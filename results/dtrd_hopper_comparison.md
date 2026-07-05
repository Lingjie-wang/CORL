# Hopper Medium DT vs DTRD-DT Comparison

Use these runs for the first comparison on `hopper-medium-v2`:

- Dense DT: `dt-hopper-medium-v2-dense-multiseed-v0`
- Sparse/Delayed DT: `dt-hopper-medium-v2-delayed-multiseed-v0`
- DTRD-DT: `dtrd-dt-hopper-medium-v2-multiseed-v0`

Aligned settings:

- Environment: `hopper-medium-v2`
- Policy model: CORL `DecisionTransformer`
- DT architecture: `embedding_dim=128`, `num_layers=3`, `num_heads=1`, `seq_len=20`
- Policy optimizer: `learning_rate=0.0008`, `weight_decay=0.0001`, `warmup_steps=10000`
- Batch size: `2048` for DTRD-DT. The original DT dense/sparse runs used
  `4096`; rerun them with `BATCH_SIZE=2048` if a strict batch-matched
  comparison is needed.
- Training steps: `100000`
- Evaluation: `eval_every=5000`, `eval_episodes=100`
- Target returns: `[3600.0, 1800.0]`
- Default seed: `10`

DTRD-specific differences:

- Adds a reward redistribution model `f_phi(s, a)`.
- Splits trajectories into train/validation sets with `val_ratio=0.2`.
- Uses `trajectory_lamb=0.01`.
- Uses `unrolled_reward_update=true` by default.

Primary metrics to compare in W&B:

- `eval/3600.0_normalized_score_mean`
- `eval/1800.0_normalized_score_mean`
- `eval/3600.0_return_mean`
- `eval/1800.0_return_mean`

If DTRD runs out of GPU memory, use:

```bash
BATCH_SIZE=1024 scripts/run_hopper_medium_dtrd_dt.sh
```

Then rerun dense/delayed DT with the same batch size before making strict claims.
