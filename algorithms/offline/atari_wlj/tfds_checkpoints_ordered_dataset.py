from typing import Iterable, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from algorithms.offline.atari_wlj.tfds_dataset import (
    _apply_reward_mode,
    _discounted_cumsum,
    _episode_steps_to_arrays,
)


CHECKPOINTS_ORDERED_VERSION = "1.1.0"
DEFAULT_NUM_CHECKPOINTS = 50


def _disable_tfds_gcs_metadata_lookup() -> None:
    # Some servers hang while TFDS probes gs://tfds-data for dataset_info.
    # The raw shards are downloaded separately and the builder reads them locally.
    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True  # pylint: disable=protected-access


def _patch_local_raw_input_prefix(raw_input_prefix: Optional[str]) -> None:
    if not raw_input_prefix:
        return
    from tensorflow_datasets.rl_unplugged.rlu_atari_checkpoints_ordered import (
        rlu_atari_checkpoints_ordered,
    )

    rlu_atari_checkpoints_ordered.RluAtariCheckpointsOrdered._INPUT_FILE_PREFIX = raw_input_prefix


def _checkpoint_split_names(
    checkpoint_splits: Optional[Union[str, Iterable[Union[str, int]]]] = None,
    num_checkpoints: int = DEFAULT_NUM_CHECKPOINTS,
) -> List[str]:
    if checkpoint_splits is None or checkpoint_splits == "" or checkpoint_splits == "all":
        return [f"checkpoint_{i:02d}" for i in range(num_checkpoints)]

    if isinstance(checkpoint_splits, str):
        raw_parts = [part.strip() for part in checkpoint_splits.split(",") if part.strip()]
    else:
        raw_parts = [str(part).strip() for part in checkpoint_splits]

    split_names: List[str] = []
    for part in raw_parts:
        if ":" in part:
            start, end = part.split(":", 1)
            start_idx = int(start) if start else 0
            end_idx = int(end) if end else num_checkpoints
            split_names.extend(f"checkpoint_{i:02d}" for i in range(start_idx, end_idx))
        elif part.startswith("checkpoint_"):
            split_names.append(part)
        else:
            split_names.append(f"checkpoint_{int(part):02d}")

    if not split_names:
        raise ValueError("checkpoint_splits did not select any splits")
    return split_names


def create_tfds_checkpoints_ordered_dataset(
    num_steps: int,
    game: str,
    data_dir: str,
    run: int = 1,
    download: bool = True,
    reward_mode: str = "dense",
    return_stepwise_returns: bool = False,
    checkpoint_splits: Optional[Union[str, Iterable[Union[str, int]]]] = None,
    raw_input_prefix: Optional[str] = None,
):
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "TFDS Atari checkpoints_ordered source requires tensorflow-datasets. "
            "Install it with: pip install tensorflow-datasets"
        ) from exc

    _disable_tfds_gcs_metadata_lookup()
    _patch_local_raw_input_prefix(raw_input_prefix)
    builder = tfds.builder(
        f"rlu_atari_checkpoints_ordered/{game}_run_{run}",
        data_dir=data_dir,
    )

    if download:
        download_config = tfds.download.DownloadConfig(try_download_gcs=False)
        builder.download_and_prepare(download_config=download_config)

    split_names = _checkpoint_split_names(
        checkpoint_splits,
        num_checkpoints=builder.num_shards(),
    )
    read_config = tfds.ReadConfig(
        shuffle_seed=0,
        try_autocache=False,
    )

    obss, actions, returns, done_idxs, stepwise_returns, timesteps = [], [], [0.0], [], [], []
    total_steps = 0

    pbar = tqdm(
        desc=f"Loading TFDS rlu_atari_checkpoints_ordered/{game}_run_{run}",
        total=num_steps,
    )
    for split_name in split_names:
        dataset = tfds.as_numpy(
            builder.as_dataset(
                split=split_name,
                read_config=read_config,
                shuffle_files=False,
            )
        )
        for example in dataset:
            frames, episode_actions, episode_rewards = _episode_steps_to_arrays(example)
            frame_stack = [
                np.zeros((84, 84), dtype=np.uint8) for _ in range(4)
            ]
            episode_len = min(len(episode_actions), num_steps - total_steps)
            if episode_len <= 0:
                break
            episode_rewards = _apply_reward_mode(
                episode_rewards[:episode_len], reward_mode
            )

            for t in range(episode_len):
                frame_stack = frame_stack[1:] + [frames[t]]
                obss.append(np.stack(frame_stack, axis=0))
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
