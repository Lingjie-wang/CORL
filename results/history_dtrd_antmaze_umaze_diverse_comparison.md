# AntMaze Umaze Diverse History-DTRD Comparison

Primary dataset:

- `antmaze-umaze-diverse-v2`

Main comparison:

- DTRD-Markov: `dtrd-markov-antmaze-umaze-diverse-v2-multiseed-v0`
- History-DTRD-GRU: `history-dtrd-gru-antmaze-umaze-diverse-v2-multiseed-v0`

Baselines:

- Original DT: `dt-antmaze-umaze-diverse-v2-original-multiseed-v0`
- Delayed DT: `dt-antmaze-umaze-diverse-v2-delayed-multiseed-v0`

Note: AntMaze is already a sparse/goal-reward benchmark. The `original`
baseline means using the D4RL rewards as provided; it is not a dense-reward
oracle.

Primary metrics:

- `eval/1.0_normalized_score_mean`
- `eval/0.5_normalized_score_mean`
- `eval/1.0_return_mean`
- `eval/0.5_return_mean`

Recommended first pass:

```bash
scripts/run_antmaze_umaze_diverse_dt_baselines.sh
scripts/run_antmaze_umaze_diverse_dtrd_markov_gru.sh
```

Run only one DTRD variant:

```bash
METHODS="gru" scripts/run_antmaze_umaze_diverse_dtrd_markov_gru.sh
METHODS="markov" scripts/run_antmaze_umaze_diverse_dtrd_markov_gru.sh
```
