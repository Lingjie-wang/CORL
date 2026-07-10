import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

from algorithms.offline.atari_dt_wlj import StateActionReturnDataset, TARGET_RETURNS
from algorithms.offline.atari_wlj.create_dataset import create_dataset
from algorithms.offline.atari_wlj.model_atari import GPT, GPTConfig
from algorithms.offline.atari_wlj.trainer_atari import Trainer, TrainerConfig
from algorithms.offline.atari_wlj.utils import set_seed


class RewardWindowDataset(Dataset):
    def __init__(self, obss, actions, stepwise_returns, done_idxs, context_length):
        self.obss = obss
        self.actions = actions
        self.stepwise_returns = stepwise_returns.astype(np.float32)
        self.done_idxs = done_idxs
        self.context_length = context_length

    def __len__(self):
        return max(1, len(self.obss) - self.context_length)

    def __getitem__(self, idx):
        done_idx = idx + self.context_length
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - self.context_length
        states = torch.tensor(
            np.array(self.obss[idx:done_idx]), dtype=torch.float32
        ).reshape(self.context_length, -1)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long)
        rewards = torch.tensor(self.stepwise_returns[idx:done_idx], dtype=torch.float32)
        return states, actions, rewards


class AtariRewardRedistributionModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
        )
        self.action_emb = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions):
        batch_size, context_length = states.shape[:2]
        state_features = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        )
        action_features = self.action_emb(actions.reshape(-1).long())
        rewards = self.head(torch.cat([state_features, action_features], dim=-1))
        return rewards.reshape(batch_size, context_length)


def discounted_cumsum(x):
    cumsum = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running += float(x[i])
        cumsum[i] = running
    return cumsum


def rtg_from_rewards(step_rewards, done_idxs):
    rtg = np.zeros_like(step_rewards, dtype=np.float32)
    start_index = 0
    for done_idx in done_idxs:
        done_idx = int(done_idx)
        rtg[start_index:done_idx] = discounted_cumsum(
            step_rewards[start_index:done_idx]
        )
        start_index = done_idx
    if start_index < len(step_rewards):
        rtg[start_index:] = discounted_cumsum(step_rewards[start_index:])
    return rtg


def train_reward_model(args, obss, actions, stepwise_returns, done_idxs, device):
    vocab_size = int(max(actions)) + 1
    dataset = RewardWindowDataset(
        obss, actions, stepwise_returns, done_idxs, args.reward_context_length
    )
    loader = DataLoader(
        dataset,
        batch_size=args.reward_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    model = AtariRewardRedistributionModel(
        vocab_size=vocab_size, hidden_dim=args.reward_hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.reward_learning_rate,
        weight_decay=args.reward_weight_decay,
    )
    for epoch in range(args.reward_epochs):
        losses = []
        pbar = trange(len(loader), desc=f"Reward epoch {epoch + 1}/{args.reward_epochs}")
        iterator = iter(loader)
        for _ in pbar:
            states, window_actions, rewards = next(iterator)
            states = states.to(device)
            window_actions = window_actions.to(device)
            rewards = rewards.to(device)
            predicted_rewards = model(states, window_actions)
            return_loss = F.mse_loss(
                predicted_rewards.sum(dim=1), rewards.sum(dim=1)
            )
            reg_loss = predicted_rewards.pow(2).mean()
            loss = return_loss + args.trajectory_lamb * reg_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.reward_clip_grad)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_description(
                f"reward epoch {epoch + 1}: loss {float(np.mean(losses)):.5f}"
            )
    return model


@torch.no_grad()
def redistribute_rewards(model, obss, actions, device, batch_size):
    model.eval()
    predicted = []
    for start in trange(0, len(obss), batch_size, desc="Redistributing rewards"):
        end = min(start + batch_size, len(obss))
        states = torch.tensor(
            np.array(obss[start:end]), dtype=torch.float32, device=device
        ).reshape(end - start, 1, -1)
        states = states / 255.0
        batch_actions = torch.tensor(
            actions[start:end], dtype=torch.long, device=device
        ).reshape(end - start, 1)
        rewards = model(states, batch_actions).reshape(-1).cpu().numpy()
        predicted.append(rewards.astype(np.float32))
    return np.concatenate(predicted, axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--context_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=500000)
    parser.add_argument("--num_buffers", type=int, default=50)
    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--trajectories_per_buffer", type=int, default=10)
    parser.add_argument("--data_dir_prefix", type=str, default="./outputs/atari/dqn_replay")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_target_return", type=int, default=None)
    parser.add_argument("--checkpoints_path", type=str, default=None)
    parser.add_argument("--reward_epochs", type=int, default=3)
    parser.add_argument("--reward_context_length", type=int, default=30)
    parser.add_argument("--reward_batch_size", type=int, default=128)
    parser.add_argument("--reward_learning_rate", type=float, default=1e-4)
    parser.add_argument("--reward_hidden_dim", type=int, default=128)
    parser.add_argument("--reward_weight_decay", type=float, default=1e-4)
    parser.add_argument("--reward_clip_grad", type=float, default=1.0)
    parser.add_argument("--trajectory_lamb", type=float, default=0.01)
    parser.add_argument("--redistribution_batch_size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    data_dir_prefix = os.path.join(args.data_dir_prefix, "")
    (
        obss,
        actions,
        returns,
        done_idxs,
        original_rtgs,
        timesteps,
        stepwise_returns,
    ) = create_dataset(
        args.num_buffers,
        args.num_steps,
        args.game,
        data_dir_prefix,
        args.trajectories_per_buffer,
        return_stepwise_returns=True,
    )

    reward_model = train_reward_model(
        args, obss, actions, stepwise_returns, done_idxs, device
    )
    redistributed_rewards = redistribute_rewards(
        reward_model, obss, actions, device, args.redistribution_batch_size
    )
    redistributed_rtgs = rtg_from_rewards(redistributed_rewards, done_idxs)

    train_dataset = StateActionReturnDataset(
        obss, args.context_length * 3, actions, done_idxs, redistributed_rtgs, timesteps
    )
    model_config = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        model_type="reward_conditioned",
        max_timestep=max(timesteps),
    )
    model = GPT(model_config)

    ckpt_path = None
    if args.checkpoints_path is not None:
        Path(args.checkpoints_path).mkdir(parents=True, exist_ok=True)
        ckpt_path = os.path.join(
            args.checkpoints_path, f"atari_dtrd_dt_{args.game}_{args.seed}.pt"
        )

    trainer_config = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.context_length * 3,
        num_workers=args.num_workers,
        seed=args.seed,
        model_type="reward_conditioned",
        game=args.game,
        max_timestep=max(timesteps),
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_target_return=args.eval_target_return or TARGET_RETURNS.get(args.game),
        ckpt_path=ckpt_path,
    )
    trainer = Trainer(model, train_dataset, None, trainer_config)
    trainer.train()
    if ckpt_path is not None:
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
