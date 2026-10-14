# Training ablations

The small JSON files in `configs/ablations/` override the shared backbone
configuration passed to `leanselect.train --config`. They use the same deployed
selector, teacher, page budget, and loss implementation as normal training.

| Override file | Change from the full selector |
|---|---|
| `full.json` | No change: four width-32 factors, learned mixing, listwise + boundary + worst-group losses |
| `direct_h1d128.json` | One width-128 factor, preserving total projected width |
| `fixed_mixing.json` | Freeze `head_mix` and `kv_head_mix` at their initialization; train both projection tensors |
| `listwise_only.json` | Set boundary and worst-group weights to zero |
| `no_boundary.json` | Set only the boundary weight to zero |
| `no_worst_group.json` | Set only the worst-group weight to zero |

These are training controls, with no bundled benchmark results. The direct
variant remains the same factorized selector with one factor; it does not
introduce an unrelated dense linear scorer. Fixed mixing retains the selector's
GQA-mean initialization and checkpoint tensor schema.

The override semantics follow the source implementation's mechanism pilots.
The public base configurations use a 2/1 teacher; the historical short pilots
used 32/16. These commands provide matched controls within the selected base
configuration, rather than reproducing those historical runs byte for byte.

For example, this single-GPU, 50-update MiniCPM pilot uses batch 8 and accumulation
8, giving an effective batch of 64. Use the same command, seed, inputs, optimizer,
and update count for each override, with a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m leanselect.train \
  --backbone minicpm --model models/MiniCPM4.1-8B \
  --data datasets/train-minicpm --config configs/minicpm.json \
  --ablation-config configs/ablations/fixed_mixing.json \
  --sequence-length 65536 --batch-size 8 --gradient-accumulation 8 \
  --steps 50 --save-every 25 --learning-rate 5e-4 \
  --output outputs/ablations/fixed_mixing
```

For NOSA, use `--backbone nosa --config configs/nosa.json` and the appropriate
model, tokenizer-matched data, and sequence length. Configurations record the
resolved override values in the training contract, so resuming with a different
ablation is rejected. Distributed runs must keep the same world size when
resuming; global batch equals world size × per-rank batch × accumulation.

For an untrained control, add `--export-initial-selector` to the fresh **full**
run. This saves `selector-000000.pt` from exactly the initialization used by that
run, before any optimizer update. Evaluate it using the same quality commands
and page budget as the trained candidates. The flag is incompatible with
`--resume`; omit it when resuming. It exports an additional checkpoint and does
not skip training.

See [reproduction.md](reproduction.md) for development-only checkpoint selection
and evaluation commands.
