from collections import deque
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm


def _discounted_cumsum(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running += float(x[i])
        out[i] = running
    return out


def _episode_steps_to_arrays(example) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = example["steps"]
    observations = steps["observation"]
    actions = steps["action"].astype(np.int64)
    rewards = steps["reward"].astype(np.float32)

    if observations.ndim == 4 and observations.shape[-1] == 1:
        observations = observations[..., 0]
    if observations.ndim != 3:
        raise ValueError(f"Expected observations shaped [T, 84, 84], got {observations.shape}")
    return observations, actions, rewards


def create_tfds_dataset(
    num_steps: int,
    game: str,
    data_dir: str,
    run: int = 1,
    download: bool = True,
    return_stepwise_returns: bool = False,
):
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "TFDS Atari source requires tensorflow-datasets. "
            "Install it with: pip install tensorflow-datasets"
        ) from exc

    builder = tfds.builder(f"rlu_atari/{game}_run_{run}", data_dir=data_dir)
    if download:
        builder.download_and_prepare()

    dataset = tfds.as_numpy(builder.as_dataset(split="train"))
    obss, actions, returns, done_idxs, stepwise_returns, timesteps = [], [], [0.0], [], [], []
    total_steps = 0

    pbar = tqdm(desc=f"Loading TFDS rlu_atari/{game}_run_{run}", total=num_steps)
    for example in dataset:
        frames, episode_actions, episode_rewards = _episode_steps_to_arrays(example)
        frame_stack = deque(
            [np.zeros((84, 84), dtype=np.uint8) for _ in range(4)], maxlen=4
        )
        episode_len = min(len(episode_actions), num_steps - total_steps)
        if episode_len <= 0:
            break

        for t in range(episode_len):
            frame_stack.append(frames[t])
            obss.append(np.stack(list(frame_stack), axis=0))
            actions.append(int(episode_actions[t]))
            reward = float(episode_rewards[t])
            stepwise_returns.append(reward)
            returns[-1] += reward
            timesteps.append(t)
            total_steps += 1
            pbar.update(1)

        done_idxs.append(len(obss))
        returns.append(0.0)
        if total_steps >= num_steps:
            break
    pbar.close()

    if returns and returns[-1] == 0.0:
        returns = returns[:-1]

    actions = np.array(actions, dtype=np.int64)
    returns = np.array(returns, dtype=np.float32)
    stepwise_returns = np.array(stepwise_returns, dtype=np.float32)
    done_idxs = np.array(done_idxs, dtype=np.int64)
    timesteps = np.array(timesteps, dtype=np.int64)

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
