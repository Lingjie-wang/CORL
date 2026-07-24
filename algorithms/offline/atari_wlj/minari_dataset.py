from __future__ import annotations

import os
from collections import deque
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

from algorithms.offline.atari_wlj.tfds_dataset import (
    _apply_reward_mode,
    _discounted_cumsum,
)

MINARI_SAMPLING_MODES = ("sequential", "balanced")


def _dataset_ids(dataset_ids: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(dataset_ids, str):
        return [part.strip() for part in dataset_ids.split(",") if part.strip()]
    return [str(part).strip() for part in dataset_ids if str(part).strip()]


def _to_84x84_frames(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)

    if observations.ndim == 3 and observations.shape[1:] == (84, 84):
        return observations.astype(np.uint8, copy=False)
    if observations.ndim == 4 and observations.shape[1:] == (84, 84, 1):
        return observations[..., 0].astype(np.uint8, copy=False)
    if observations.ndim == 4 and observations.shape[-1] == 3:
        frames = []
        for obs in observations:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            frames.append(cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA))
        return np.asarray(frames, dtype=np.uint8)
    if observations.ndim == 3 and observations.shape[1:] == (210, 160):
        frames = [
            cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
            for obs in observations
        ]
        return np.asarray(frames, dtype=np.uint8)

    raise ValueError(f"Unsupported Minari observation shape: {observations.shape}")


def create_minari_dataset(
    num_steps: int,
    dataset_ids: Union[str, Iterable[str]],
    data_dir: Optional[str] = None,
    reward_mode: str = "dense",
    download: bool = False,
    return_stepwise_returns: bool = False,
    sampling_mode: str = "balanced",
    sampling_seed: int = 0,
) -> Tuple:
    try:
        import minari
    except ImportError as exc:
        raise ImportError(
            "Minari Atari source requires minari. Install it with: "
            "pip install 'minari[hdf5,hf]' gymnasium ale-py"
        ) from exc

    if data_dir:
        os.environ["MINARI_DATASETS_PATH"] = os.path.abspath(data_dir)

    ids = _dataset_ids(dataset_ids)
    if not ids:
        raise ValueError("No Minari dataset ids were provided")
    if sampling_mode not in MINARI_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported Minari sampling_mode={sampling_mode}. "
            f"Use one of {MINARI_SAMPLING_MODES}."
        )

    datasets = [minari.load_dataset(dataset_id, download=download) for dataset_id in ids]
    total_available_steps = sum(int(dataset.total_steps) for dataset in datasets)
    if total_available_steps < num_steps:
        raise ValueError(
            f"Requested {num_steps} steps from Minari datasets {ids}, "
            f"but only {total_available_steps} are available."
        )

    obss, actions, returns, done_idxs, stepwise_returns, timesteps = [], [], [0.0], [], [], []
    total_steps = 0
    pbar = tqdm(desc=f"Loading Minari datasets {','.join(ids)}", total=num_steps)
    if sampling_mode == "balanced":
        base_steps = num_steps // len(datasets)
        remainder = num_steps % len(datasets)
        dataset_step_targets = [
            base_steps + (1 if i < remainder else 0)
            for i in range(len(datasets))
        ]
    else:
        dataset_step_targets = [num_steps] * len(datasets)

    print(
        "Minari sampling:",
        f"mode={sampling_mode}",
        f"datasets={len(datasets)}",
        f"step_targets={sorted(set(dataset_step_targets))}",
    )

    for dataset_idx, dataset in enumerate(datasets):
        dataset_steps = 0
        dataset_target = dataset_step_targets[dataset_idx]
        if dataset_target <= 0:
            continue
        episode_indices = np.arange(dataset.total_episodes)
        if sampling_mode == "balanced":
            rng = np.random.RandomState(sampling_seed + dataset_idx)
            rng.shuffle(episode_indices)
        else:
            episode_indices = None

        for episode in dataset.iterate_episodes(episode_indices):
            episode_actions = np.asarray(episode.actions, dtype=np.int64)
            episode_rewards = np.asarray(episode.rewards, dtype=np.float32)
            frames = _to_84x84_frames(episode.observations)
            episode_len = min(
                len(episode_actions),
                len(episode_rewards),
                len(frames) - 1 if len(frames) == len(episode_actions) + 1 else len(frames),
                num_steps - total_steps,
                dataset_target - dataset_steps,
            )
            if episode_len <= 0:
                continue

            episode_rewards = _apply_reward_mode(
                episode_rewards[:episode_len],
                reward_mode,
            )
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
                dataset_steps += 1
                pbar.update(1)

            done_idxs.append(len(obss))
            returns.append(0.0)
            if total_steps >= num_steps or dataset_steps >= dataset_target:
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
