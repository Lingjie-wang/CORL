import argparse
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset

from algorithms.offline.atari_wlj.hdf5_atari_dataset import HDF5_SAMPLING_MODES
from algorithms.offline.atari_wlj.model_atari import GPT, GPTConfig
from algorithms.offline.atari_wlj.minari_dataset import MINARI_SAMPLING_MODES
from algorithms.offline.atari_wlj.tfds_checkpoints_ordered_dataset import (
    SAMPLING_MODES,
    create_tfds_checkpoints_ordered_dataset as create_tfds_dataset,
)
from algorithms.offline.atari_wlj.trainer_atari import Trainer, TrainerConfig
from algorithms.offline.atari_wlj.utils import set_seed


# Eval reward-conditioning targets. Set to the DTRD paper values (global_game_info)
# so the DT baseline and the DTRD method are evaluated under the SAME target
# return. Seaquest/Qbert differ from the original Decision Transformer values
# (1150/14000); Breakout/Pong are unchanged.
TARGET_RETURNS = {
    "Breakout": 90,
    "Seaquest": 290,
    "Qbert": 662,
    "Pong": 20,
}


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = int(max(actions)) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        context_length = self.block_size // 3
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
        timesteps = torch.tensor(
            self.timesteps[idx : idx + 1], dtype=torch.int64
        ).unsqueeze(1)
        return states, actions, rtgs, timesteps


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
    # wandb params
    parser.add_argument("--project", type=str, default="CORL")
    parser.add_argument("--group", type=str, default="DT-Atari")
    parser.add_argument("--name", type=str, default="DT")
    parser.add_argument(
        "--wandb_suffix",
        type=str,
        default="",
        help="Optional suffix appended to generated wandb group/name, e.g. target290.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--context_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="reward_conditioned")
    parser.add_argument("--num_steps", type=int, default=500000)
    parser.add_argument("--num_buffers", type=int, default=50)
    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--trajectories_per_buffer", type=int, default=10)
    parser.add_argument("--data_dir_prefix", type=str, default="./outputs/atari/dqn_replay")
    parser.add_argument(
        "--data_source",
        choices=("dqn_replay", "tfds", "minari", "hdf5"),
        default="tfds",
    )
    parser.add_argument("--reward_mode", choices=("dense", "delayed", "sparse"), default="dense")
    parser.add_argument("--tfds_data_dir", type=str, default="./data/atari/tfds_checkpoints_ordered")
    parser.add_argument("--tfds_run", type=int, default=1)
    parser.add_argument("--tfds_checkpoint_splits", type=str, default="all")
    parser.add_argument("--tfds_sampling_mode", choices=SAMPLING_MODES, default="sequential")
    parser.add_argument("--tfds_sampling_seed", type=int, default=None)
    parser.add_argument("--tfds_raw_input_prefix", type=str, default=None)
    parser.add_argument("--tfds_download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minari_dataset_id", type=str, default=None)
    parser.add_argument("--minari_data_dir", type=str, default="./data/minari")
    parser.add_argument("--minari_download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--minari_num_shards", type=int, default=50)
    parser.add_argument("--minari_dataset_prefix", type=str, default=None)
    parser.add_argument("--minari_sampling_mode", choices=MINARI_SAMPLING_MODES, default="balanced")
    parser.add_argument("--minari_sampling_seed", type=int, default=None)
    parser.add_argument("--hdf5_data_dir", type=str, default="./data/atari/dqn_replay_hdf5")
    parser.add_argument("--hdf5_shard_paths", type=str, default=None)
    parser.add_argument("--hdf5_num_shards", type=int, default=50)
    parser.add_argument("--hdf5_sampling_mode", choices=HDF5_SAMPLING_MODES, default="balanced")
    parser.add_argument("--hdf5_sampling_seed", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_target_return", type=int, default=None)
    parser.add_argument(
        "--eval_rtg_update",
        choices=("dense", "clipped_dense", "delayed"),
        default=None,
        help=(
            "How to update RTG during online eval. By default, dense reward "
            "runs subtract the environment reward each step, clipped_dense "
            "subtracts reward clipped to [-1, 1], while sparse/delayed reward "
            "runs keep RTG constant until episode end."
        ),
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="If set, also evaluate every N training steps in addition to once per epoch",
    )
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
    if args.eval_rtg_update is None:
        args.eval_rtg_update = (
            "delayed" if args.reward_mode in ("sparse", "delayed") else "dense"
        )
    if args.tfds_sampling_seed is None:
        args.tfds_sampling_seed = args.seed
    if args.minari_sampling_seed is None:
        args.minari_sampling_seed = args.seed
    if args.hdf5_sampling_seed is None:
        args.hdf5_sampling_seed = args.seed

    # init wandb session for logging. group by game+reward_mode so that the
    # dense/delayed x multi-seed runs are aggregated together, and give each
    # run a unique, descriptive name.
    wandb_config = vars(args).copy()
    eval_suffix = "" if args.eval_rtg_update == "dense" else f"-eval-{args.eval_rtg_update}"
    sampling_suffix = (
        "" if args.data_source != "tfds" or args.tfds_sampling_mode == "sequential"
        else f"-{args.tfds_sampling_mode}"
    )
    model_suffix = "" if args.model_type == "reward_conditioned" else f"-{args.model_type}"
    run_suffix = f"-{args.wandb_suffix.strip('-')}" if args.wandb_suffix else ""
    wandb_config["group"] = (
        f"{args.group}-{args.game}-{args.reward_mode}{model_suffix}"
        f"{sampling_suffix}{eval_suffix}{run_suffix}"
    )
    wandb_config["name"] = (
        f"{args.name}-{args.game}-{args.reward_mode}{model_suffix}"
        f"{sampling_suffix}{eval_suffix}{run_suffix}-{args.seed}-{str(uuid.uuid4())[:8]}"
    )
    wandb_init(wandb_config)

    if args.data_source == "tfds":
        obss, actions, returns, done_idxs, rtgs, timesteps = create_tfds_dataset(
            args.num_steps,
            args.game,
            args.tfds_data_dir,
            run=args.tfds_run,
            download=args.tfds_download,
            reward_mode=args.reward_mode,
            checkpoint_splits=args.tfds_checkpoint_splits,
            raw_input_prefix=args.tfds_raw_input_prefix,
            sampling_mode=args.tfds_sampling_mode,
            sampling_seed=args.tfds_sampling_seed,
            trajectories_per_buffer=args.trajectories_per_buffer,
        )
    elif args.data_source == "minari":
        from algorithms.offline.atari_wlj.minari_dataset import create_minari_dataset

        if args.minari_dataset_id:
            minari_dataset_id = args.minari_dataset_id
        else:
            minari_prefix = (
                args.minari_dataset_prefix
                or f"corl/{args.game.lower()}-dqn-epoch"
            )
            minari_dataset_id = ",".join(
                f"{minari_prefix}-{shard_idx:02d}-v0"
                for shard_idx in range(1, args.minari_num_shards + 1)
            )
        obss, actions, returns, done_idxs, rtgs, timesteps = create_minari_dataset(
            args.num_steps,
            minari_dataset_id,
            data_dir=args.minari_data_dir,
            reward_mode=args.reward_mode,
            download=args.minari_download,
            sampling_mode=args.minari_sampling_mode,
            sampling_seed=args.minari_sampling_seed,
        )
    elif args.data_source == "hdf5":
        from algorithms.offline.atari_wlj.hdf5_atari_dataset import (
            create_hdf5_atari_dataset,
        )

        obss, actions, returns, done_idxs, rtgs, timesteps = create_hdf5_atari_dataset(
            args.num_steps,
            args.game,
            args.hdf5_data_dir,
            reward_mode=args.reward_mode,
            shard_paths=args.hdf5_shard_paths,
            num_shards=args.hdf5_num_shards,
            sampling_mode=args.hdf5_sampling_mode,
            sampling_seed=args.hdf5_sampling_seed,
        )
    else:
        from algorithms.offline.atari_wlj.create_dataset import create_dataset

        data_dir_prefix = os.path.join(args.data_dir_prefix, "")
        obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(
            args.num_buffers,
            args.num_steps,
            args.game,
            data_dir_prefix,
            args.trajectories_per_buffer,
            reward_mode=args.reward_mode,
        )
    train_dataset = StateActionReturnDataset(
        obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps
    )

    model_config = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        model_type=args.model_type,
        max_timestep=max(timesteps),
    )
    model = GPT(model_config)

    ckpt_path = None
    if args.checkpoints_path is not None:
        Path(args.checkpoints_path).mkdir(parents=True, exist_ok=True)
        ckpt_path = os.path.join(
            args.checkpoints_path, f"atari_dt_{args.game}_{args.seed}.pt"
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
        model_type=args.model_type,
        game=args.game,
        max_timestep=max(timesteps),
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_target_return=args.eval_target_return or TARGET_RETURNS.get(args.game),
        eval_rtg_update=args.eval_rtg_update,
        eval_every_steps=args.eval_every_steps,
        ckpt_path=ckpt_path,
    )
    trainer = Trainer(model, train_dataset, None, trainer_config)
    trainer.train()
    if ckpt_path is not None:
        trainer.save_checkpoint()
    wandb.finish()


if __name__ == "__main__":
    main()
