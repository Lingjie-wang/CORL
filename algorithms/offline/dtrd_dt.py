# CORL adaptation of DTRD for continuous-control D4RL tasks.
# Paper: Towards Long-delayed Sparsity: Learning a Better Transformer
# through Reward Redistribution.
import os
import random
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa

from algorithms.offline.dt import (
    DecisionTransformer,
    discounted_cumsum,
    eval_rollout,
    pad_along_axis,
    set_seed,
    wrap_env,
)


@dataclass
class TrainConfig:
    project: str = "CORL"
    group: str = "DTRD-DT-D4RL"
    name: str = "DTRD-DT"

    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0

    env_name: str = "hopper-medium-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4

    reward_learning_rate: float = 1e-4
    reward_hidden_dim: int = 256
    reward_num_layers: int = 2
    reward_model_type: str = "markov"
    reward_history_len: int = 0
    trajectory_lamb: float = 0.01
    unrolled_reward_update: bool = True
    val_ratio: float = 0.2

    target_returns: Tuple[float, ...] = (3600.0, 1800.0)
    eval_episodes: int = 100
    eval_every: int = 5_000

    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1)")
        if self.reward_model_type not in ("markov", "gru"):
            raise ValueError("reward_model_type must be either 'markov' or 'gru'")
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    trajectories, traj_lens, traj_returns = [], [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            episode_data["dense_returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            episode_data["trajectory_return"] = np.array(
                episode_data["rewards"].sum(), dtype=np.float32
            )
            episode_data["sparse_returns"] = np.full_like(
                episode_data["rewards"], episode_data["trajectory_return"]
            )
            trajectories.append(episode_data)
            traj_lens.append(episode_data["actions"].shape[0])
            traj_returns.append(float(episode_data["trajectory_return"]))
            data_ = defaultdict(list)

    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_lens),
        "traj_returns": np.array(traj_returns),
    }
    return trajectories, info


def split_trajectories(
    trajectories: List[DefaultDict[str, np.ndarray]],
    traj_lens: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[List[DefaultDict[str, np.ndarray]], List[DefaultDict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(trajectories))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:val_size].tolist())
    train, val = [], []
    for i, traj in enumerate(trajectories):
        (val if i in val_indices else train).append(traj)
    if not train:
        train.append(val.pop())
    return train, val


class DTRDSequenceDataset(IterableDataset):
    def __init__(
        self,
        trajectories: List[DefaultDict[str, np.ndarray]],
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        seq_len: int,
        episode_len: int,
        reward_scale: float,
        reward_history_len: int = 0,
    ):
        self.dataset = trajectories
        self.state_mean = obs_mean
        self.state_std = obs_std
        self.seq_len = seq_len
        self.episode_len = episode_len
        self.reward_scale = reward_scale
        self.reward_history_len = reward_history_len
        self.history_pad_len = episode_len if reward_history_len == 0 else reward_history_len
        self.history_pad_len = max(self.history_pad_len, seq_len)
        self.sample_prob = np.array([t["actions"].shape[0] for t in trajectories])
        self.sample_prob = self.sample_prob / self.sample_prob.sum()

    def __prepare_sample(self, traj_idx: int, start_idx: int):
        traj = self.dataset[traj_idx]
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )

        states = (states - self.state_mean) / self.state_std
        end_idx = min(start_idx + self.seq_len, traj["actions"].shape[0])
        history_start = max(0, end_idx - self.history_pad_len)
        history_states = traj["observations"][history_start:end_idx]
        history_actions = traj["actions"][history_start:end_idx]
        history_states = (history_states - self.state_mean) / self.state_std
        history_mask = np.hstack(
            [
                np.ones(history_states.shape[0]),
                np.zeros(self.history_pad_len - history_states.shape[0]),
            ]
        )
        history_context_start = start_idx - history_start

        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
        if history_states.shape[0] < self.history_pad_len:
            history_states = pad_along_axis(history_states, pad_to=self.history_pad_len)
            history_actions = pad_along_axis(
                history_actions, pad_to=self.history_pad_len
            )

        return (
            states.astype(np.float32),
            actions.astype(np.float32),
            history_states.astype(np.float32),
            history_actions.astype(np.float32),
            history_mask.astype(np.float32),
            np.array(history_context_start, dtype=np.int64),
            time_steps.astype(np.int64),
            mask.astype(np.float32),
            np.array(float(traj["trajectory_return"]), dtype=np.float32),
            np.array(traj["actions"].shape[0], dtype=np.float32),
            np.array(start_idx, dtype=np.float32),
        )

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            traj_len = self.dataset[traj_idx]["actions"].shape[0]
            start_idx = random.randint(0, traj_len - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


class RewardRedistributionModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)

    def get_context_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        history_mask: torch.Tensor,
        history_context_starts: torch.Tensor,
    ) -> torch.Tensor:
        return self(states, actions)


class HistoryGRURewardRedistributionModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([history_states, history_actions], dim=-1)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        rewards = self.reward_head(out).squeeze(-1)
        return rewards * history_mask

    def get_context_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        history_mask: torch.Tensor,
        history_context_starts: torch.Tensor,
    ) -> torch.Tensor:
        history_rewards = self(history_states, history_actions, history_mask)
        seq_len = states.shape[1]
        offsets = torch.arange(seq_len, device=states.device).unsqueeze(0)
        gather_idx = history_context_starts.long().unsqueeze(1) + offsets
        gather_idx = gather_idx.clamp(max=history_rewards.shape[1] - 1)
        return torch.gather(history_rewards, dim=1, index=gather_idx)


def dtrd_returns_and_regularizer(
    reward_model: RewardRedistributionModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    history_states: torch.Tensor,
    history_actions: torch.Tensor,
    history_mask: torch.Tensor,
    history_context_starts: torch.Tensor,
    mask: torch.Tensor,
    traj_returns: torch.Tensor,
    traj_lens: torch.Tensor,
    start_idxs: torch.Tensor,
    reward_scale: float,
    trajectory_lamb: float,
    detach_reward: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step_rewards = reward_model.get_context_rewards(
        states,
        actions,
        history_states,
        history_actions,
        history_mask,
        history_context_starts,
    )
    step_rewards = step_rewards * mask
    if detach_reward:
        step_rewards = step_rewards.detach()

    past_baseline = traj_returns * start_idxs / traj_lens.clamp_min(1.0)
    elapsed_in_context = torch.cumsum(step_rewards, dim=1) - step_rewards
    redistributed_returns = (
        traj_returns.unsqueeze(1) - past_baseline.unsqueeze(1) - elapsed_in_context
    )
    redistributed_returns = redistributed_returns * reward_scale

    valid_lens = mask.sum(dim=1).clamp_min(1.0)
    scaled_context_sum = step_rewards.sum(dim=1) * traj_lens / valid_lens
    regularizer = (
        (scaled_context_sum - traj_returns).pow(2).mean()
        * trajectory_lamb
        * reward_scale
    )
    return redistributed_returns, regularizer, step_rewards


def concat_tensors(xs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([x.reshape(-1) for x in xs])


def zero_like_grads(
    grads: Tuple[Optional[torch.Tensor], ...], params: List[torch.nn.Parameter]
) -> List[torch.Tensor]:
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def policy_action_loss(
    model: DecisionTransformer,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    time_steps: torch.Tensor,
    padding_mask: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    predicted_actions = model(
        states=states,
        actions=actions,
        returns_to_go=returns,
        time_steps=time_steps,
        padding_mask=padding_mask,
    )
    loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
    return (loss * mask.unsqueeze(-1)).mean()


class DTRDGradientApproximation:
    def __init__(
        self,
        model: DecisionTransformer,
        reward_model: RewardRedistributionModel,
        eps: float = 0.01,
    ):
        self.new_model = deepcopy(model)
        self.reward_model = reward_model
        self.eps = eps

    @staticmethod
    def sync_model(source: nn.Module, target: nn.Module) -> None:
        target.load_state_dict(source.state_dict())

    def compute_unrolled_model(
        self,
        model: DecisionTransformer,
        train_loss: torch.Tensor,
        eta: float,
        weight_decay: float,
    ) -> None:
        model_params = list(model.parameters())
        grads = torch.autograd.grad(
            train_loss,
            model_params,
            allow_unused=True,
            retain_graph=True,
        )
        grads = zero_like_grads(grads, model_params)
        with torch.no_grad():
            for param, new_param, grad in zip(
                model_params, self.new_model.parameters(), grads
            ):
                new_param.copy_(param - eta * (grad + weight_decay * param))

    def hessian_vector_product(
        self,
        model: DecisionTransformer,
        vector: List[torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        vector_norm = concat_tensors(vector).norm()
        if vector_norm.item() == 0.0:
            return [torch.zeros_like(p) for p in self.reward_model.parameters()]
        radius = self.eps / vector_norm

        with torch.no_grad():
            for param, vec in zip(model.parameters(), vector):
                param.add_(radius * vec)
        loss_p = policy_action_loss(
            model, states, actions, returns, time_steps, padding_mask, mask
        )
        reward_params = list(self.reward_model.parameters())
        grads_p = torch.autograd.grad(
            loss_p,
            reward_params,
            allow_unused=True,
            retain_graph=True,
        )
        grads_p = zero_like_grads(grads_p, reward_params)

        with torch.no_grad():
            for param, vec in zip(model.parameters(), vector):
                param.sub_(2.0 * radius * vec)
        loss_n = policy_action_loss(
            model, states, actions, returns, time_steps, padding_mask, mask
        )
        grads_n = torch.autograd.grad(
            loss_n,
            reward_params,
            allow_unused=True,
            retain_graph=True,
        )
        grads_n = zero_like_grads(grads_n, reward_params)

        with torch.no_grad():
            for param, vec in zip(model.parameters(), vector):
                param.add_(radius * vec)

        return [(gp - gn) / (2.0 * radius) for gp, gn in zip(grads_p, grads_n)]

    def reward_step(
        self,
        model: DecisionTransformer,
        train_loss: torch.Tensor,
        val_loss_regularizer: torch.Tensor,
        eta: float,
        weight_decay: float,
        train_states: torch.Tensor,
        train_actions: torch.Tensor,
        train_returns: torch.Tensor,
        train_time_steps: torch.Tensor,
        train_padding_mask: torch.Tensor,
        train_mask: torch.Tensor,
        val_states: torch.Tensor,
        val_actions: torch.Tensor,
        val_returns: torch.Tensor,
        val_time_steps: torch.Tensor,
        val_padding_mask: torch.Tensor,
        val_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.compute_unrolled_model(model, train_loss, eta, weight_decay)
        val_loss = policy_action_loss(
            self.new_model,
            val_states,
            val_actions,
            val_returns,
            val_time_steps,
            val_padding_mask,
            val_mask,
        )

        self.new_model.zero_grad()
        self.reward_model.zero_grad()
        val_loss.backward(retain_graph=True)
        vector = [
            torch.zeros_like(param) if param.grad is None else param.grad.detach()
            for param in self.new_model.parameters()
        ]
        direct_grads = [
            torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
            for param in self.reward_model.parameters()
        ]

        hvp = self.hessian_vector_product(
            model,
            vector,
            train_states,
            train_actions,
            train_returns,
            train_time_steps,
            train_padding_mask,
            train_mask,
        )

        self.reward_model.zero_grad()
        val_loss_regularizer.backward()
        with torch.no_grad():
            for param, direct_grad, hvp_grad in zip(
                self.reward_model.parameters(), direct_grads, hvp
            ):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.add_(direct_grad - eta * hvp_grad)

        return val_loss, val_loss + val_loss_regularizer


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    trajectories, info = load_d4rl_trajectories(config.env_name, gamma=1.0)
    train_traj, val_traj = split_trajectories(
        trajectories, info["traj_lens"], config.val_ratio, config.train_seed
    )

    train_dataset = DTRDSequenceDataset(
        train_traj,
        info["obs_mean"],
        info["obs_std"],
        config.seq_len,
        config.episode_len,
        config.reward_scale,
        config.reward_history_len,
    )
    val_dataset = DTRDSequenceDataset(
        val_traj,
        info["obs_mean"],
        info["obs_std"],
        config.seq_len,
        config.episode_len,
        config.reward_scale,
        config.reward_history_len,
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=train_dataset.state_mean,
        state_std=train_dataset.state_std,
        reward_scale=config.reward_scale,
    )
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]

    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)
    reward_model_cls = (
        RewardRedistributionModel
        if config.reward_model_type == "markov"
        else HistoryGRURewardRedistributionModel
    )
    reward_model = reward_model_cls(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.reward_hidden_dim,
        num_layers=config.reward_num_layers,
    ).to(config.device)
    dtrd_gradient = DTRDGradientApproximation(model, reward_model)
    dtrd_gradient.new_model.to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    reward_optim = torch.optim.Adam(
        reward_model.parameters(), lr=config.reward_learning_rate
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda steps: min((steps + 1) / config.warmup_steps, 1)
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Policy parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Reward parameters: {sum(p.numel() for p in reward_model.parameters())}")

    train_iter = iter(trainloader)
    val_iter = iter(valloader)
    for step in trange(config.update_steps, desc="Training DTRD-DT"):
        batch = next(train_iter)
        (
            states,
            actions,
            history_states,
            history_actions,
            history_mask,
            history_context_starts,
            time_steps,
            mask,
            traj_returns,
            traj_lens,
            start_idxs,
        ) = [b.to(config.device) for b in batch]
        padding_mask = ~mask.to(torch.bool)

        val_batch = next(val_iter)
        (
            v_states,
            v_actions,
            v_history_states,
            v_history_actions,
            v_history_mask,
            v_history_context_starts,
            v_time_steps,
            v_mask,
            v_returns,
            v_lens,
            v_starts,
        ) = [b.to(config.device) for b in val_batch]
        v_padding_mask = ~v_mask.to(torch.bool)

        reward_train_returns, _, _ = dtrd_returns_and_regularizer(
            reward_model,
            states,
            actions,
            history_states,
            history_actions,
            history_mask,
            history_context_starts,
            mask,
            traj_returns,
            traj_lens,
            start_idxs,
            config.reward_scale,
            config.trajectory_lamb,
            detach_reward=not config.unrolled_reward_update,
        )
        reward_returns, reward_reg, reward_steps = dtrd_returns_and_regularizer(
            reward_model,
            v_states,
            v_actions,
            v_history_states,
            v_history_actions,
            v_history_mask,
            v_history_context_starts,
            v_mask,
            v_returns,
            v_lens,
            v_starts,
            config.reward_scale,
            config.trajectory_lamb,
            detach_reward=False,
        )

        if config.unrolled_reward_update:
            train_loss_for_unroll = policy_action_loss(
                model,
                states,
                actions,
                reward_train_returns,
                time_steps,
                padding_mask,
                mask,
            )
            lr = scheduler.get_last_lr()[0]
            reward_val_loss, reward_loss = dtrd_gradient.reward_step(
                model,
                train_loss_for_unroll,
                reward_reg,
                lr,
                config.weight_decay,
                states,
                actions,
                reward_train_returns,
                time_steps,
                padding_mask,
                mask,
                v_states,
                v_actions,
                reward_returns,
                v_time_steps,
                v_padding_mask,
                v_mask,
            )
        else:
            for param in model.parameters():
                param.requires_grad_(False)
            reward_val_loss = policy_action_loss(
                model,
                v_states,
                v_actions,
                reward_returns,
                v_time_steps,
                v_padding_mask,
                v_mask,
            )
            for param in model.parameters():
                param.requires_grad_(True)
            reward_loss = reward_val_loss + reward_reg
            reward_optim.zero_grad()
            reward_loss.backward()

        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), config.clip_grad)
        reward_optim.step()

        returns, _, _ = dtrd_returns_and_regularizer(
            reward_model,
            states,
            actions,
            history_states,
            history_actions,
            history_mask,
            history_context_starts,
            mask,
            traj_returns,
            traj_lens,
            start_idxs,
            config.reward_scale,
            config.trajectory_lamb,
            detach_reward=True,
        )
        policy_loss = policy_action_loss(
            model,
            states,
            actions,
            returns,
            time_steps,
            padding_mask,
            mask,
        )

        optim.zero_grad()
        policy_loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train/policy_loss": policy_loss.item(),
                "train/reward_val_loss": reward_val_loss.item(),
                "train/reward_regularizer": reward_reg.item(),
                "train/reward_loss": reward_loss.item(),
                "train/reward_mean": (reward_steps * v_mask).sum().item()
                / v_mask.sum().item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "reward_learning_rate": reward_optim.param_groups[0]["lr"],
            },
            step=step,
        )

        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, _ = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        reward_mode="dense",
                        device=config.device,
                    )
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                    eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "reward_model_state": reward_model.state_dict(),
            "state_mean": train_dataset.state_mean,
            "state_std": train_dataset.state_std,
        }
        torch.save(
            checkpoint, os.path.join(config.checkpoints_path, "dtrd_dt_checkpoint.pt")
        )

    wandb.finish()


if __name__ == "__main__":
    train()
