#!/usr/bin/env python
"""Train DQN and store Atari replay-style epoch shards as raw uint8 HDF5."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing
from tqdm.auto import tqdm


class NatureDQN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float() / 255.0)


class DQNReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, int],
        stack_size: int = 4,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.stack_size = int(stack_size)
        self.rng = np.random.default_rng(seed)
        self.observations = np.empty(
            (self.capacity, *observation_shape), dtype=np.uint8
        )
        self.next_observations = np.empty(
            (self.capacity, *observation_shape), dtype=np.uint8
        )
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.bool_)
        self.add_count = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.add_count % self.capacity
        self.observations[idx] = observation
        self.next_observations[idx] = next_observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.add_count += 1
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    @property
    def first_abs_idx(self) -> int:
        return self.add_count - self.size

    def _slot(self, abs_idx: int) -> int:
        return int(abs_idx % self.capacity)

    def _stack(self, abs_idx: int) -> np.ndarray:
        frames = np.zeros(
            (self.stack_size, *self.observations.shape[1:]), dtype=np.uint8
        )
        first = self.first_abs_idx
        for pos in range(self.stack_size):
            frame_abs_idx = abs_idx - self.stack_size + 1 + pos
            if frame_abs_idx < first:
                continue
            crosses_done = False
            for boundary_abs_idx in range(frame_abs_idx, abs_idx):
                if self.dones[self._slot(boundary_abs_idx)]:
                    crosses_done = True
                    break
            if not crosses_done:
                frames[pos] = self.observations[self._slot(frame_abs_idx)]
        return frames

    def sample(self, batch_size: int, device: torch.device):
        abs_indices = self.rng.integers(
            self.first_abs_idx,
            self.add_count,
            size=batch_size,
            endpoint=False,
        )
        states = np.stack([self._stack(int(i)) for i in abs_indices], axis=0)
        next_states = states.copy()
        for row, abs_idx in enumerate(abs_indices):
            next_states[row, :-1] = states[row, 1:]
            next_states[row, -1] = self.next_observations[self._slot(int(abs_idx))]

        slots = np.asarray([self._slot(int(i)) for i in abs_indices])
        return (
            torch.as_tensor(states, device=device),
            torch.as_tensor(self.actions[slots], device=device, dtype=torch.long),
            torch.as_tensor(self.rewards[slots], device=device, dtype=torch.float32),
            torch.as_tensor(next_states, device=device),
            torch.as_tensor(self.dones[slots], device=device, dtype=torch.float32),
        )


class RawHDF5ShardWriter:
    def __init__(
        self,
        path: Path,
        num_steps: int,
        observation_shape: tuple[int, int],
        game: str,
        epoch: int,
        seed: int,
        compression: str | None = "lzf",
        flush_every: int = 8192,
        overwrite: bool = False,
    ):
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.num_steps = int(num_steps)
        self.flush_every = int(flush_every)
        self.index = 0
        self.current_return = 0.0
        self.current_length = 0
        self.episode_ends: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []

        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"Raw HDF5 shard already exists: {self.path}. "
                "Use --overwrite or choose another --data_dir."
            )
        if self.path.exists() and overwrite:
            self.path.unlink()
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.file = h5py.File(self.tmp_path, "w")
        self.file.attrs["format"] = "corl_atari_dqn_replay_hdf5"
        self.file.attrs["format_version"] = "1.0"
        self.file.attrs["game"] = game
        self.file.attrs["epoch"] = int(epoch)
        self.file.attrs["seed"] = int(seed)
        self.file.attrs["observation_encoding"] = "raw_uint8"
        self.file.attrs["lossless"] = True
        self.file.attrs["num_steps"] = self.num_steps

        obs_chunks = (min(1024, self.num_steps), *observation_shape)
        vector_chunks = (min(8192, self.num_steps),)
        self.observations = self.file.create_dataset(
            "observations",
            shape=(self.num_steps, *observation_shape),
            dtype=np.uint8,
            chunks=obs_chunks,
            compression=compression,
        )
        self.actions = self.file.create_dataset(
            "actions",
            shape=(self.num_steps,),
            dtype=np.int64,
            chunks=vector_chunks,
            compression=compression,
        )
        self.rewards = self.file.create_dataset(
            "rewards",
            shape=(self.num_steps,),
            dtype=np.float32,
            chunks=vector_chunks,
            compression=compression,
        )
        self.terminals = self.file.create_dataset(
            "terminals",
            shape=(self.num_steps,),
            dtype=np.bool_,
            chunks=vector_chunks,
            compression=compression,
        )
        self.truncations = self.file.create_dataset(
            "truncations",
            shape=(self.num_steps,),
            dtype=np.bool_,
            chunks=vector_chunks,
            compression=compression,
        )

        self.obs_buffer = np.empty(
            (self.flush_every, *observation_shape), dtype=np.uint8
        )
        self.action_buffer = np.empty((self.flush_every,), dtype=np.int64)
        self.reward_buffer = np.empty((self.flush_every,), dtype=np.float32)
        self.terminal_buffer = np.empty((self.flush_every,), dtype=np.bool_)
        self.truncation_buffer = np.empty((self.flush_every,), dtype=np.bool_)
        self.buffer_size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        terminal: bool,
        truncated: bool,
    ) -> None:
        if self.index >= self.num_steps:
            raise RuntimeError("Shard writer received more steps than configured")
        buf_idx = self.buffer_size
        self.obs_buffer[buf_idx] = observation
        self.action_buffer[buf_idx] = action
        self.reward_buffer[buf_idx] = reward
        self.terminal_buffer[buf_idx] = terminal
        self.truncation_buffer[buf_idx] = truncated
        self.buffer_size += 1
        self.index += 1

        self.current_return += float(reward)
        self.current_length += 1
        if terminal or truncated:
            self._finish_episode()
        if self.buffer_size >= self.flush_every:
            self.flush()

    def _finish_episode(self) -> None:
        if self.current_length <= 0:
            return
        self.episode_ends.append(self.index)
        self.episode_returns.append(self.current_return)
        self.episode_lengths.append(self.current_length)
        self.current_return = 0.0
        self.current_length = 0

    def flush(self) -> None:
        if self.buffer_size <= 0:
            return
        end = self.index
        start = end - self.buffer_size
        self.observations[start:end] = self.obs_buffer[: self.buffer_size]
        self.actions[start:end] = self.action_buffer[: self.buffer_size]
        self.rewards[start:end] = self.reward_buffer[: self.buffer_size]
        self.terminals[start:end] = self.terminal_buffer[: self.buffer_size]
        self.truncations[start:end] = self.truncation_buffer[: self.buffer_size]
        self.buffer_size = 0

    def close(self, commit: bool) -> None:
        if self.file is None:
            return
        if self.current_length > 0:
            if self.buffer_size > 0:
                self.truncation_buffer[self.buffer_size - 1] = True
            else:
                self.truncations[self.index - 1] = True
            self._finish_episode()
        self.flush()
        self.file.create_dataset(
            "episode_ends", data=np.asarray(self.episode_ends, dtype=np.int64)
        )
        self.file.create_dataset(
            "episode_returns",
            data=np.asarray(self.episode_returns, dtype=np.float32),
        )
        self.file.create_dataset(
            "episode_lengths",
            data=np.asarray(self.episode_lengths, dtype=np.int64),
        )
        self.file.attrs["written_steps"] = int(self.index)
        self.file.close()
        self.file = None
        if commit:
            self.tmp_path.rename(self.path)


@dataclass
class EpisodeStats:
    count: int = 0
    current_return: float = 0.0
    current_length: int = 0
    returns: list[float] | None = None
    lengths: list[int] | None = None

    def __post_init__(self) -> None:
        if self.returns is None:
            self.returns = []
        if self.lengths is None:
            self.lengths = []

    def step(self, reward: float) -> None:
        self.current_return += float(reward)
        self.current_length += 1

    def finish(self) -> None:
        assert self.returns is not None
        assert self.lengths is not None
        self.returns.append(self.current_return)
        self.lengths.append(self.current_length)
        self.count += 1
        self.current_return = 0.0
        self.current_length = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Seaquest")
    parser.add_argument("--data_dir", type=str, default="./data/atari/dqn_replay_hdf5")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--replay_capacity", type=int, default=1_000_000)
    parser.add_argument("--learning_starts", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--target_update_interval", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--rmsprop_alpha", type=float, default=0.95)
    parser.add_argument("--rmsprop_eps", type=float, default=0.01)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay_steps", type=int, default=1_000_000)
    parser.add_argument("--repeat_action_probability", type=float, default=0.0)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--clip_training_reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hdf5_compression", type=str, default="lzf")
    parser.add_argument("--flush_every", type=int, default=8192)
    parser.add_argument("--model_dir", type=str, default="./outputs/atari/hdf5_dqn")
    parser.add_argument("--log_every_steps", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_env(args: argparse.Namespace, epoch: int):
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(
        f"ALE/{args.game}-v5",
        obs_type="grayscale",
        frameskip=1,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=False,
    )
    env.action_space.seed(args.seed + epoch)
    return AtariPreprocessing(
        env,
        noop_max=args.noop_max,
        frame_skip=args.frame_skip,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )


def epsilon_at(step: int, args: argparse.Namespace) -> float:
    progress = min(1.0, step / max(1, args.epsilon_decay_steps))
    return args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)


def reset_stack(observation: np.ndarray) -> deque:
    return deque(
        [np.zeros_like(observation) for _ in range(3)] + [observation],
        maxlen=4,
    )


def select_action(
    model: NatureDQN,
    frames: deque,
    epsilon: float,
    num_actions: int,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(num_actions))
    state = np.stack(list(frames), axis=0)[None]
    with torch.no_grad():
        q_values = model(torch.as_tensor(state, device=device))
    return int(torch.argmax(q_values, dim=1).item())


def update_dqn(
    model: NatureDQN,
    target_model: NatureDQN,
    optimizer: torch.optim.Optimizer,
    replay: DQNReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    states, actions, rewards, next_states, dones = replay.sample(args.batch_size, device)
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_model(next_states).max(dim=1).values
        targets = rewards + args.gamma * (1.0 - dones) * next_q_values
    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    data_dir = Path(args.data_dir).expanduser().resolve() / args.game
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir).expanduser().resolve() / args.game
    model_dir.mkdir(parents=True, exist_ok=True)

    probe_env = make_env(args, epoch=0)
    observation, _ = probe_env.reset(seed=args.seed)
    num_actions = int(probe_env.action_space.n)
    observation_shape = tuple(observation.shape)
    probe_env.close()
    if observation_shape != (84, 84):
        raise ValueError(f"Expected 84x84 observations, got {observation_shape}")

    model = NatureDQN(num_actions).to(device)
    target_model = NatureDQN(num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        alpha=args.rmsprop_alpha,
        eps=args.rmsprop_eps,
    )
    replay = DQNReplayBuffer(
        capacity=args.replay_capacity,
        observation_shape=observation_shape,
        stack_size=4,
        seed=args.seed,
    )

    global_step = 0
    recent_losses: deque[float] = deque(maxlen=100)
    for epoch in range(1, args.num_epochs + 1):
        shard_path = data_dir / f"epoch_{epoch:02d}.hdf5"
        env = make_env(args, epoch=epoch)
        writer = RawHDF5ShardWriter(
            shard_path,
            num_steps=args.steps_per_epoch,
            observation_shape=observation_shape,
            game=args.game,
            epoch=epoch,
            seed=args.seed,
            compression=None if args.hdf5_compression == "none" else args.hdf5_compression,
            flush_every=args.flush_every,
            overwrite=args.overwrite,
        )
        observation, _ = env.reset(seed=args.seed + epoch)
        frames = reset_stack(observation)
        stats = EpisodeStats()
        epoch_steps = 0
        pbar = tqdm(
            total=args.steps_per_epoch,
            desc=f"Collecting raw HDF5 {args.game} epoch_{epoch:02d}",
            dynamic_ncols=True,
        )
        commit = False
        try:
            while epoch_steps < args.steps_per_epoch:
                epsilon = epsilon_at(global_step, args)
                action = select_action(
                    model,
                    frames,
                    epsilon,
                    num_actions,
                    device,
                    rng,
                )
                next_observation, raw_reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                train_reward = (
                    float(np.clip(raw_reward, -1.0, 1.0))
                    if args.clip_training_reward
                    else float(raw_reward)
                )
                replay.add(observation, action, train_reward, next_observation, done)
                writer.add(observation, action, float(raw_reward), terminated, truncated)

                frames.append(next_observation)
                stats.step(float(raw_reward))
                global_step += 1
                epoch_steps += 1
                pbar.update(1)

                if (
                    global_step >= args.learning_starts
                    and len(replay) >= args.batch_size
                    and global_step % args.train_frequency == 0
                ):
                    loss = update_dqn(model, target_model, optimizer, replay, args, device)
                    recent_losses.append(loss)

                if global_step % args.target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

                if done:
                    stats.finish()
                    observation, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    frames = reset_stack(observation)
                else:
                    observation = next_observation

                if args.log_every_steps and global_step % args.log_every_steps == 0:
                    mean_return = (
                        float(np.mean(stats.returns[-20:]))
                        if stats.returns
                        else float("nan")
                    )
                    mean_loss = (
                        float(np.mean(recent_losses))
                        if recent_losses
                        else float("nan")
                    )
                    pbar.set_postfix(
                        epsilon=f"{epsilon:.3f}",
                        replay=len(replay),
                        return20=f"{mean_return:.1f}",
                        loss=f"{mean_loss:.4f}",
                    )

            commit = True
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "target_model": target_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(checkpoint, model_dir / f"dqn_{args.game}_epoch_{epoch:02d}.pt")
        finally:
            writer.close(commit=commit)
            pbar.close()
            env.close()
            if not commit and shard_path.with_suffix(shard_path.suffix + ".tmp").exists():
                print(f"Kept incomplete shard temp file: {shard_path}.tmp")

        returns = stats.returns or []
        print(
            f"Saved {shard_path}: steps={args.steps_per_epoch}, "
            f"episodes={len(writer.episode_ends)}, "
            f"global_step={global_step}, "
            f"raw_return_mean={np.mean(returns) if returns else float('nan'):.2f}, "
            f"raw_return_max={np.max(returns) if returns else float('nan'):.2f}"
        )

    print("Finished raw HDF5 DQN replay collection.")
    print(f"Data dir={data_dir}")


if __name__ == "__main__":
    main()
