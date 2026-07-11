import argparse
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset

from algorithms.offline.atari_wlj.model_atari import GPT, GPTConfig
from algorithms.offline.atari_wlj.tfds_dataset import create_tfds_dataset
from algorithms.offline.atari_wlj.trainer_atari import Trainer, TrainerConfig
from algorithms.offline.atari_wlj.utils import set_seed


TARGET_RETURNS = {
    "Breakout": 90,
    "Seaquest": 1150,
    "Qbert": 14000,
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
    parser.add_argument("--data_source", choices=("dqn_replay", "tfds"), default="dqn_replay")
    parser.add_argument("--reward_mode", choices=("dense", "delayed", "sparse"), default="dense")
    parser.add_argument("--tfds_data_dir", type=str, default="./outputs/atari/tfds")
    parser.add_argument("--tfds_run", type=int, default=1)
    parser.add_argument("--tfds_download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_target_return", type=int, default=None)
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

    # init wandb session for logging. group by game+reward_mode so that the
    # dense/delayed x multi-seed runs are aggregated together, and give each
    # run a unique, descriptive name.
    wandb_config = vars(args).copy()
    wandb_config["group"] = f"{args.group}-{args.game}-{args.reward_mode}"
    wandb_config["name"] = (
        f"{args.name}-{args.game}-{args.reward_mode}-{args.seed}-{str(uuid.uuid4())[:8]}"
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
        ckpt_path=ckpt_path,
    )
    trainer = Trainer(model, train_dataset, None, trainer_config)
    trainer.train()
    if ckpt_path is not None:
        trainer.save_checkpoint()
    wandb.finish()


if __name__ == "__main__":
    main()
