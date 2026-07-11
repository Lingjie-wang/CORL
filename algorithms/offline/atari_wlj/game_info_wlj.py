"""Per-game configuration for DTRD reward redistribution.

Ported verbatim from the official DTRD repository (mingpt/utils.py
``global_game_info``) so that the faithful reproduction uses the exact same
per-game action space, reward categorisation and evaluation target return as
the paper "Towards Long-delayed Sparsity: Learning a Better Transformer through
Reward Redistribution" (Zhu et al., IJCAI 2023).

Notes:
- ``target_reward`` is the return the Decision Transformer is conditioned on at
  evaluation time. These are the DTRD paper values (Breakout=90, Seaquest=290,
  Qbert=662, Pong=20), NOT the original Decision Transformer values.
- ``reward_vector`` / ``reward_category_num`` drive the discrete redistribution
  network (gumbel-softmax over reward categories).
- ``reward_range`` drives the continuous redistribution network (tanh/sigmoid
  rescaled into [min_reward, max_reward]).
"""

global_game_info = {
    "Breakout": {
        "action_dim": 4,
        "reward_range": {"min_reward": 0.0, "max_reward": 1.0},
        "reward_vector": [0.0, 1.0],
        "reward_category_num": 2,
        "target_reward": 90.0,
    },
    "Seaquest": {
        "action_dim": 18,
        "reward_range": {"min_reward": 0.0, "max_reward": 1.0},
        "reward_vector": [0.0, 1.0],
        "reward_category_num": 2,
        "target_reward": 290.0,
    },
    "Qbert": {
        "action_dim": 6,
        "reward_range": {"min_reward": 0.0, "max_reward": 1.0},
        "reward_vector": [0.0, 1.0],
        "reward_category_num": 2,
        "target_reward": 662.0,
    },
    "Pong": {
        "action_dim": 6,
        "reward_range": {"min_reward": -1.0, "max_reward": 1.0},
        "reward_vector": [-1.0, 0.0, 1.0],
        "reward_category_num": 3,
        "target_reward": 20.0,
    },
}
