"""Faithful reproduction of DTRD (Decision Transformer + Reward Redistribution)
on CORL, following "Towards Long-delayed Sparsity: Learning a Better Transformer
through Reward Redistribution" (Zhu et al., IJCAI 2023).

Unlike the simplified two-stage ``atari_dtrd_dt_wlj.py`` (which trains a reward
model with MSE then trains a plain DT), this entry point reproduces the paper's
mechanism: the reward-redistribution network and the Decision Transformer policy
are trained JOINTLY per batch via a DARTS-style bilevel meta-gradient, and at
evaluation the return-to-go is decremented by the LEARNED redistributed reward.

Reproduction assumptions (documented, logged to wandb config):
- The official repo ships pre-split train/val ``.npz`` files; the split method
  is not published. We hold out the last ``--val_fraction`` of trajectories as
  the validation set used by the bilevel meta-objective.
- Reward-conditioning uses the sparse (delayed) rtgs, so this entry point forces
  ``reward_mode='delayed'`` for the dataset.
"""

import argparse
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset

from algorithms.offline.atari_wlj.dtrd_trainer_wlj import DTRDModelTrainer, DTRDTrainerConfig
from algorithms.offline.atari_wlj.game_info_wlj import global_game_info
from algorithms.offline.atari_wlj.model_atari import GPT, GPTConfig
from algorithms.offline.atari_wlj.reward_redistribute_wlj import (
    ContinuousRedistributeNetwork,
    DiscreteRedistributeNetwork,
    RedistributeConfig,
    TrajectoryDataset,
)
from algorithms.offline.atari_wlj.tfds_checkpoints_ordered_dataset import create_tfds_checkpoints_ordered_dataset as create_tfds_dataset
from algorithms.offline.atari_wlj.utils import set_seed


class StateActionReturnDataset(Dataset):
    """Like the baseline DT dataset but also returns a per-window trajectory
    index (needed by TrajectoryDataset to look up head rtgs / traj length).

    ``rtgs`` here are the sparse (delayed) rtgs used for conditioning.
    """

    def __init__(self, obss, context_length, actions, done_idxs, rtgs, timesteps, trajectory_index):
        self.context_length = context_length
        self.vocab_size = int(max(actions)) + 1
        self.data = obss
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.trajectory_index = trajectory_index

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        context_length = self.context_length
        done_idx = idx + context_length
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - context_length
        states = torch.tensor(
            np.array(self.data[idx:done_idx]), dtype=torch.float32
        ).reshape(context_length, -1)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)
        trajectory_index = torch.tensor(
            self.trajectory_index[idx:done_idx], dtype=torch.long
        ).unsqueeze(1)
        return states, actions, rtgs, timesteps, trajectory_index


def _slice_by_trajectories(obss, actions, rtgs, timesteps, done_idxs, traj_lo, traj_hi):
    """Extract trajectories [traj_lo, traj_hi) into a self-contained split.

    Returns rebased arrays plus a 0-based per-step ``trajectory_index`` and
    rebased ``done_idxs`` (offsets into the returned split), so the split can
    drive its own TrajectoryDataset.
    """
    starts = np.concatenate([[0], done_idxs[:-1]])  # start offset of each traj
    step_lo = int(starts[traj_lo])
    step_hi = int(done_idxs[traj_hi - 1])

    split_obss = obss[step_lo:step_hi]
    split_actions = actions[step_lo:step_hi]
    split_rtgs = rtgs[step_lo:step_hi]
    split_timesteps = timesteps[step_lo:step_hi]

    split_done_idxs = (done_idxs[traj_lo:traj_hi] - step_lo).astype(np.int64)
    # per-step trajectory id, 0-based within this split
    traj_index = np.zeros(step_hi - step_lo, dtype=np.int64)
    prev = 0
    for local_id, done in enumerate(split_done_idxs):
        traj_index[prev:int(done)] = local_id
        prev = int(done)
    return split_obss, split_actions, split_rtgs, split_timesteps, split_done_idxs, traj_index


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # wandb
    parser.add_argument("--project", type=str, default="CORL")
    parser.add_argument("--group", type=str, default="DTRD-Atari")
    parser.add_argument("--name", type=str, default="DTRD")
    # basic
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--context_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--num_steps", type=int, default=500000)
    parser.add_argument("--num_buffers", type=int, default=50)
    parser.add_argument("--trajectories_per_buffer", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of trajectories held out as the bilevel val set")
    # data
    parser.add_argument("--data_dir_prefix", type=str, default="./outputs/atari/dqn_replay")
    parser.add_argument("--data_source", choices=("dqn_replay", "tfds"), default="tfds")
    parser.add_argument("--tfds_data_dir", type=str, default="./data/atari/tfds_checkpoints_ordered")
    parser.add_argument("--tfds_run", type=int, default=1)
    parser.add_argument("--tfds_checkpoint_splits", type=str, default="all")
    parser.add_argument("--tfds_raw_input_prefix", type=str, default=None)
    parser.add_argument("--tfds_download", action=argparse.BooleanOptionalAction, default=True)
    # policy optim
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=128)
    # redistribution
    parser.add_argument("--discrete_redistribute", type=int, default=0)
    parser.add_argument("--redistribute_learning_rate", type=float, default=1e-3)
    parser.add_argument("--redistribute_step_size", type=int, default=1000)
    parser.add_argument("--redistribute_gamma", type=float, default=0.9)
    parser.add_argument("--trajectory_lamb", type=float, default=1e-2)
    # eval / ckpt
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_eval_steps", type=int, default=27000,
                        help="Hard per-episode step cap during eval (prevents runaway rollouts)")
    parser.add_argument("--eval_target_return", type=float, default=None)
    parser.add_argument("--checkpoints_path", type=str, default=None)
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
    game_info = global_game_info[args.game]

    # wandb: faithful reproduction records the reproduction assumptions too
    wandb_config = vars(args).copy()
    wandb_config["method"] = "DTRD-faithful-bilevel"
    wandb_config["reward_mode"] = "delayed"
    wandb_config["group"] = f"{args.group}-{args.game}"
    wandb_config["name"] = f"{args.name}-{args.game}-{args.seed}-{str(uuid.uuid4())[:8]}"
    wandb_init(wandb_config)

    # Load data (reward_mode forced to 'delayed' -> rtg is the sparse rtg)
    if args.data_source == "tfds":
        obss, actions, returns, done_idxs, rtgs, timesteps, _ = create_tfds_dataset(
            args.num_steps, args.game, args.tfds_data_dir, run=args.tfds_run,
            download=args.tfds_download, reward_mode="delayed", return_stepwise_returns=True,
            checkpoint_splits=args.tfds_checkpoint_splits,
            raw_input_prefix=args.tfds_raw_input_prefix,
        )
    else:
        from algorithms.offline.atari_wlj.create_dataset import create_dataset

        data_dir_prefix = os.path.join(args.data_dir_prefix, "")
        obss, actions, returns, done_idxs, rtgs, timesteps, _ = create_dataset(
            args.num_buffers, args.num_steps, args.game, data_dir_prefix,
            args.trajectories_per_buffer, reward_mode="delayed", return_stepwise_returns=True,
        )

    # Train / val split by trajectory (last val_fraction held out for the meta-objective)
    n_traj = len(done_idxs)
    n_val = max(1, int(round(n_traj * args.val_fraction)))
    n_train = max(1, n_traj - n_val)
    max_timestep = int(max(timesteps))

    tr = _slice_by_trajectories(obss, actions, rtgs, timesteps, done_idxs, 0, n_train)
    va = _slice_by_trajectories(obss, actions, rtgs, timesteps, done_idxs, n_train, n_traj)
    (tr_obss, tr_actions, tr_rtgs, tr_ts, tr_done, tr_tidx) = tr
    (va_obss, va_actions, va_rtgs, va_ts, va_done, va_tidx) = va

    train_dataset = StateActionReturnDataset(
        tr_obss, args.context_length, tr_actions, tr_done, tr_rtgs, tr_ts, tr_tidx
    )
    val_dataset = StateActionReturnDataset(
        va_obss, args.context_length, va_actions, va_done, va_rtgs, va_ts, va_tidx
    )
    train_traj = TrajectoryDataset(
        tr_obss, tr_actions, tr_rtgs, tr_done, args.discrete_redistribute, args.trajectory_lamb
    )
    val_traj = TrajectoryDataset(
        va_obss, va_actions, va_rtgs, va_done, args.discrete_redistribute, args.trajectory_lamb
    )

    # Policy (Decision Transformer). block_size = context_length*3 for the
    # (rtg, state, action) interleaving in CORL's GPT.
    model_config = GPTConfig(
        game_info["action_dim"], args.context_length * 3,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        model_type="reward_conditioned", max_timestep=max_timestep,
    )
    model = GPT(model_config)

    # Reward redistribution network (discrete or continuous, per game)
    rconf = RedistributeConfig(
        action_dim=game_info["action_dim"], n_embd=args.n_embd, device=device,
        context_length=args.context_length, redistribute_activate_func="tanh",
        reward_category_num=game_info["reward_category_num"],
        reward_vector=game_info["reward_vector"], reward_range=game_info["reward_range"],
    )
    redistribute = (
        DiscreteRedistributeNetwork(rconf) if args.discrete_redistribute
        else ContinuousRedistributeNetwork(rconf)
    ).to(device)

    ckpt_path = None
    if args.checkpoints_path is not None:
        Path(args.checkpoints_path).mkdir(parents=True, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoints_path, f"atari_dtrd_{args.game}_{args.seed}.pt")

    trainer_config = DTRDTrainerConfig(
        max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        lr_decay=True, warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.context_length * 3,
        num_workers=args.num_workers, seed=args.seed, game=args.game,
        max_timestep=max_timestep, device=args.device, context_length=args.context_length,
        redistribute_learning_rate=args.redistribute_learning_rate,
        redistribute_step_size=args.redistribute_step_size,
        redistribute_gamma=args.redistribute_gamma,
        eval_episodes=args.eval_episodes,
        max_eval_steps=args.max_eval_steps,
        eval_target_return=args.eval_target_return or game_info["target_reward"],
        ckpt_path=ckpt_path,
    )
    trainer = DTRDModelTrainer(
        model=model, redistribute=redistribute, device=device,
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_trajectory_dataset=train_traj, val_trajectory_dataset=val_traj,
        config=trainer_config,
    )
    trainer.train()
    if ckpt_path is not None:
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
