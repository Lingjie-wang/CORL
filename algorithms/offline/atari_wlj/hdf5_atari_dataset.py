from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm.auto import tqdm

from algorithms.offline.atari_wlj.tfds_dataset import (
    _apply_reward_mode,
    _discounted_cumsum,
)


HDF5_SAMPLING_MODES = ("sequential", "balanced")


def _split_csv(values: Union[str, Iterable[str], None]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [part.strip() for part in values.split(",") if part.strip()]
    return [str(part).strip() for part in values if str(part).strip()]


def default_hdf5_shard_paths(
    data_dir: str,
    game: str,
    num_shards: int = 50,
) -> list[str]:
    root = Path(data_dir).expanduser()
    return [
        str(root / game / f"epoch_{shard_idx:02d}.hdf5")
        for shard_idx in range(1, num_shards + 1)
    ]


def _resolve_shard_paths(
    data_dir: str,
    game: str,
    shard_paths: Union[str, Iterable[str], None],
    num_shards: int,
) -> list[str]:
    explicit_paths = _split_csv(shard_paths)
    if explicit_paths:
        return explicit_paths
    return default_hdf5_shard_paths(data_dir, game, num_shards)


def _episode_ranges(f: h5py.File) -> list[tuple[int, int]]:
    if "episode_ends" in f:
        ends = np.asarray(f["episode_ends"], dtype=np.int64)
    else:
        terminals = np.asarray(f["terminals"], dtype=np.bool_)
        truncations = np.asarray(f.get("truncations", np.zeros_like(terminals)), dtype=np.bool_)
        ends = np.flatnonzero(np.logical_or(terminals, truncations)) + 1
        if len(ends) == 0 or ends[-1] != len(terminals):
            ends = np.r_[ends, len(terminals)]

    ranges = []
    start = 0
    for end in ends:
        end = int(end)
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def create_hdf5_atari_dataset(
    num_steps: int,
    game: str,
    data_dir: str,
    reward_mode: str = "dense",
    return_stepwise_returns: bool = False,
    shard_paths: Union[str, Iterable[str], None] = None,
    num_shards: int = 50,
    sampling_mode: str = "balanced",
    sampling_seed: int = 0,
) -> Tuple:
    if sampling_mode not in HDF5_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported HDF5 sampling_mode={sampling_mode}. "
            f"Use one of {HDF5_SAMPLING_MODES}."
        )

    paths = _resolve_shard_paths(data_dir, game, shard_paths, num_shards)
    missing_paths = [path for path in paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing raw HDF5 Atari shard(s):\n"
            + "\n".join(missing_paths[:10])
            + ("\n..." if len(missing_paths) > 10 else "")
        )

    with h5py.File(paths[0], "r") as f:
        if f["observations"].dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 observations in {paths[0]}, "
                f"got {f['observations'].dtype}"
            )
        if tuple(f["observations"].shape[1:]) != (84, 84):
            raise ValueError(
                f"Expected observations shaped [N,84,84] in {paths[0]}, "
                f"got {f['observations'].shape}"
            )

    total_available_steps = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            total_available_steps += int(f["actions"].shape[0])
    if total_available_steps < num_steps:
        raise ValueError(
            f"Requested {num_steps} steps from raw HDF5 shards, "
            f"but only {total_available_steps} are available."
        )

    if sampling_mode == "balanced":
        base_steps = num_steps // len(paths)
        remainder = num_steps % len(paths)
        shard_step_targets = [
            base_steps + (1 if i < remainder else 0)
            for i in range(len(paths))
        ]
    else:
        shard_step_targets = [num_steps] * len(paths)

    print(
        "Raw HDF5 Atari sampling:",
        f"mode={sampling_mode}",
        f"shards={len(paths)}",
        f"step_targets={sorted(set(shard_step_targets))}",
    )

    obss, actions, returns, done_idxs, stepwise_returns, timesteps = [], [], [0.0], [], [], []
    total_steps = 0
    pbar = tqdm(desc=f"Loading raw HDF5 Atari {game}", total=num_steps)

    for shard_idx, (path, shard_target) in enumerate(zip(paths, shard_step_targets)):
        if shard_target <= 0:
            continue
        shard_steps = 0
        with h5py.File(path, "r") as f:
            episode_ranges = _episode_ranges(f)
            if sampling_mode == "balanced":
                rng = np.random.RandomState(sampling_seed + shard_idx)
                rng.shuffle(episode_ranges)

            observations = f["observations"]
            shard_actions = f["actions"]
            shard_rewards = f["rewards"]
            for start, end in episode_ranges:
                episode_len = min(
                    end - start,
                    num_steps - total_steps,
                    shard_target - shard_steps,
                )
                if episode_len <= 0:
                    continue

                frames = np.asarray(observations[start : start + episode_len], dtype=np.uint8)
                episode_actions = np.asarray(
                    shard_actions[start : start + episode_len],
                    dtype=np.int64,
                )
                episode_rewards = np.asarray(
                    shard_rewards[start : start + episode_len],
                    dtype=np.float32,
                )
                episode_rewards = _apply_reward_mode(episode_rewards, reward_mode)

                frame_stack = deque(
                    [np.zeros((84, 84), dtype=np.uint8) for _ in range(4)],
                    maxlen=4,
                )
                for t in range(episode_len):
                    frame_stack.append(frames[t])
                    obss.append(np.stack(list(frame_stack), axis=0))
                    actions.append(int(episode_actions[t]))
                    reward = float(episode_rewards[t])
                    stepwise_returns.append(reward)
                    returns[-1] += reward
                    timesteps.append(t)
                    total_steps += 1
                    shard_steps += 1
                    pbar.update(1)

                done_idxs.append(len(obss))
                returns.append(0.0)
                if total_steps >= num_steps or shard_steps >= shard_target:
                    break
        if total_steps >= num_steps:
            break
    pbar.close()

    if returns and returns[-1] == 0.0:
        returns = returns[:-1]

    actions = np.asarray(actions, dtype=np.int64)
    returns = np.asarray(returns, dtype=np.float32)
    stepwise_returns = np.asarray(stepwise_returns, dtype=np.float32)
    done_idxs = np.asarray(done_idxs, dtype=np.int64)
    timesteps = np.asarray(timesteps, dtype=np.int64)

    rtg = np.zeros_like(stepwise_returns, dtype=np.float32)
    start_index = 0
    for done_idx in done_idxs:
        done_idx = int(done_idx)
        rtg[start_index:done_idx] = _discounted_cumsum(
            stepwise_returns[start_index:done_idx]
        )
        start_index = done_idx

    if return_stepwise_returns:
        return obss, actions, returns, done_idxs, rtg, timesteps, stepwise_returns
    return obss, actions, returns, done_idxs, rtg, timesteps
