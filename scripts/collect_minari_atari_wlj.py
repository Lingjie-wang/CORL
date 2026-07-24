#!/usr/bin/env python
"""Collect Atari rollouts into a local Minari dataset.

This records raw ALE rewards by default. Observations are preprocessed to
84x84 grayscale frames so CORL can frame-stack them at load time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import minari
import numpy as np
from gymnasium.wrappers import AtariPreprocessing
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Seaquest")
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--minari_data_dir", type=str, default="./data/minari")
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("random", "noop"), default="random")
    parser.add_argument("--checkpoint_every_episodes", type=int, default=50)
    parser.add_argument("--repeat_action_probability", type=float, default=0.0)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--data_format", type=str, default=None)
    parser.add_argument("--author", type=str, default="yewei")
    parser.add_argument("--author_email", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_env(args: argparse.Namespace):
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(
        f"ALE/{args.game}-v5",
        obs_type="grayscale",
        frameskip=1,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=False,
    )
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


def choose_action(env, policy: str) -> int:
    if policy == "noop":
        return 0
    if policy == "random":
        return int(env.action_space.sample())
    raise ValueError(f"Unsupported policy={policy}")


def main() -> None:
    args = parse_args()
    dataset_id = args.dataset_id or f"corl/{args.game.lower()}-{args.policy}-v0"
    minari_root = Path(args.minari_data_dir).expanduser().resolve()
    os.environ["MINARI_DATASETS_PATH"] = str(minari_root)
    minari_root.mkdir(parents=True, exist_ok=True)

    dataset_path = minari_root / dataset_id
    if dataset_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Minari dataset already exists: {dataset_path}. "
            "Use --overwrite or choose another --dataset_id."
        )
    if dataset_path.exists() and args.overwrite:
        import shutil

        shutil.rmtree(dataset_path)

    env = minari.DataCollector(
        make_env(args),
        record_infos=False,
        data_format=args.data_format,
    )
    rng = np.random.default_rng(args.seed)
    dataset = None
    episode_count = 0
    total_steps = 0
    episode_return = 0.0
    returns = []

    obs, _ = env.reset(seed=args.seed)
    del obs
    pbar = tqdm(total=args.total_steps, desc=f"Collecting {dataset_id}")
    try:
        while total_steps < args.total_steps:
            action = choose_action(env, args.policy)
            obs, reward, terminated, truncated, _ = env.step(action)
            del obs
            total_steps += 1
            episode_return += float(reward)
            pbar.update(1)

            if terminated or truncated:
                returns.append(episode_return)
                episode_count += 1
                episode_return = 0.0
                if total_steps < args.total_steps:
                    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    del obs

                if (
                    args.checkpoint_every_episodes > 0
                    and episode_count % args.checkpoint_every_episodes == 0
                ):
                    if dataset is None:
                        dataset = env.create_dataset(
                            dataset_id=dataset_id,
                            algorithm_name=f"{args.policy}-policy",
                            author=args.author,
                            author_email=args.author_email,
                            description=(
                                f"{args.game} Atari rollouts collected with Minari. "
                                "Rewards are raw ALE rewards; observations are "
                                "84x84 uint8 grayscale frames."
                            ),
                            requirements=[
                                "minari==0.5.3",
                                "gymnasium",
                                "ale-py",
                            ],
                        )
                    else:
                        env.add_to_dataset(dataset)

        if dataset is None:
            dataset = env.create_dataset(
                dataset_id=dataset_id,
                algorithm_name=f"{args.policy}-policy",
                author=args.author,
                author_email=args.author_email,
                description=(
                    f"{args.game} Atari rollouts collected with Minari. "
                    "Rewards are raw ALE rewards; observations are "
                    "84x84 uint8 grayscale frames."
                ),
                requirements=["minari==0.5.3", "gymnasium", "ale-py"],
            )
        else:
            env.add_to_dataset(dataset)
    finally:
        pbar.close()
        env.close()

    print(f"Saved Minari dataset: {dataset_id}")
    print(f"MINARI_DATASETS_PATH={minari_root}")
    print(f"Total steps: {dataset.total_steps}")
    print(f"Total episodes: {dataset.total_episodes}")
    if returns:
        print(
            "Episode raw return:",
            f"mean={np.mean(returns):.2f}",
            f"min={np.min(returns):.2f}",
            f"max={np.max(returns):.2f}",
        )


if __name__ == "__main__":
    main()
