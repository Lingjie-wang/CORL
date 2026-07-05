# AntMaze Medium Diverse History-DTRD Comparison

Primary dataset:

- `antmaze-medium-diverse-v2`

Main comparison:

- DTRD-Markov: `dtrd-markov-antmaze-medium-diverse-v2-multiseed-v0`
- History-DTRD-GRU: `history-dtrd-gru-antmaze-medium-diverse-v2-multiseed-v0`

Baselines:

- Original DT: `dt-antmaze-medium-diverse-v2-original-multiseed-v0`
- Delayed DT: `dt-antmaze-medium-diverse-v2-delayed-multiseed-v0`

Note: AntMaze is already a sparse/goal-reward benchmark. The `original`
baseline means using the D4RL rewards as provided; it is not a dense-reward
oracle. The `delayed` baseline moves each trajectory's total reward to the
last timestep.

Primary metrics:

- `eval/1.0_normalized_score_mean`
- `eval/0.5_normalized_score_mean`
- `eval/1.0_return_mean`
- `eval/0.5_return_mean`

Recommended first pass:

```bash
# Baselines
scripts/run_antmaze_medium_diverse_dt_baselines.sh

# DTRD variants
scripts/run_antmaze_medium_diverse_dtrd_markov_gru.sh
```

For a cheaper smoke run:

```bash
UPDATE_STEPS=10000 EVAL_EVERY=2000 EVAL_EPISODES=10 scripts/run_antmaze_medium_diverse_dtrd_markov_gru.sh
```
