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
SAMPLING_MODES = ("sequential", "balanced", "dt_replay")


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


def _split_step_targets(num_steps: int, num_splits: int) -> List[int]:
    base_steps = num_steps // num_splits
    remainder = num_steps % num_splits
    return [
        base_steps + (1 if split_idx < remainder else 0)
        for split_idx in range(num_splits)
    ]


def _iter_shuffled_split_examples(builder, split_name: str, read_config, tfds, rng):
    num_examples = builder.info.splits[split_name].num_examples
    episode_indices = np.arange(num_examples)
    rng.shuffle(episode_indices)

    for episode_idx in episode_indices:
        dataset = tfds.as_numpy(
            builder.as_dataset(
                split=f"{split_name}[{int(episode_idx)}:{int(episode_idx) + 1}]",
                read_config=read_config,
                shuffle_files=False,
            )
        )
        yield from dataset


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
    sampling_mode: str = "sequential",
    sampling_seed: int = 0,
    trajectories_per_buffer: int = 10,
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

    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unsupported sampling_mode={sampling_mode}. Use one of {SAMPLING_MODES}."
        )

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
    split_step_targets = (
        _split_step_targets(num_steps, len(split_names))
        if sampling_mode == "balanced"
        else [num_steps] * len(split_names)
    )
    print(
        "TFDS checkpoint sampling:",
        f"mode={sampling_mode}",
        f"splits={len(split_names)}",
        f"step_targets={sorted(set(split_step_targets))}",
        f"trajectories_per_buffer={trajectories_per_buffer}",
    )

    def append_example(example, max_episode_steps: int) -> int:
        nonlocal total_steps

        frames, episode_actions, episode_rewards = _episode_steps_to_arrays(example)
        frame_stack = [
            np.zeros((84, 84), dtype=np.uint8) for _ in range(4)
        ]
        episode_len = min(
            len(episode_actions),
            num_steps - total_steps,
            max_episode_steps,
        )
        if episode_len <= 0:
            return 0
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
        return episode_len

    if sampling_mode == "dt_replay":
        rng = np.random.RandomState(sampling_seed)
        split_iters = {}

        def get_split_iter(split_name):
            if split_name not in split_iters:
                split_iters[split_name] = iter(
                    tfds.as_numpy(
                        builder.as_dataset(
                            split=split_name,
                            read_config=read_config,
                            shuffle_files=False,
                        )
                    )
                )
            return split_iters[split_name]

        exhausted_splits = set()
        split_steps = {split_name: 0 for split_name in split_names}
        while total_steps < num_steps and len(exhausted_splits) < len(split_names):
            available_splits = [
                split_name for split_name in split_names
                if split_name not in exhausted_splits
            ]
            split_name = available_splits[int(rng.randint(len(available_splits)))]
            loaded_trajectories = 0
            while (
                total_steps < num_steps
                and loaded_trajectories < trajectories_per_buffer
            ):
                try:
                    example = next(get_split_iter(split_name))
                except StopIteration:
                    exhausted_splits.add(split_name)
                    break
                episode_steps = append_example(example, num_steps - total_steps)
                if episode_steps > 0:
                    split_steps[split_name] += episode_steps
                    loaded_trajectories += 1
        print(
            "TFDS dt_replay split step stats:",
            f"min={min(split_steps.values()) if split_steps else 0}",
            f"max={max(split_steps.values()) if split_steps else 0}",
        )
    else:
        for split_idx, (split_name, split_step_target) in enumerate(
            zip(split_names, split_step_targets)
        ):
            if split_step_target <= 0:
                continue
            if sampling_mode == "balanced":
                rng = np.random.RandomState(sampling_seed + split_idx)
                dataset = _iter_shuffled_split_examples(
                    builder, split_name, read_config, tfds, rng
                )
            else:
                dataset = tfds.as_numpy(
                    builder.as_dataset(
                        split=split_name,
                        read_config=read_config,
                        shuffle_files=False,
                    )
                )
            split_steps = 0
            for example in dataset:
                episode_steps = append_example(
                    example, split_step_target - split_steps
                )
                split_steps += episode_steps
                if (
                    total_steps >= num_steps
                    or split_steps >= split_step_target
                ):
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
