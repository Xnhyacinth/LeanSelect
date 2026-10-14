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

The public base configurations use a 2/1 teacher. The latest matched campaign
below adds a 32/16 teacher-recipe contrast and three training seeds per backbone.
The optional short pilot shown later uses the same controls but is not the
300-update campaign.

## Matched 12-run campaign

`leanselect.ablation_campaign` generates the native distributed training command
for each latest campaign run. It only prints a plan: it does not import GPU
modules, create output directories, or submit training. Supply the selected
backbone's local model and tokenizer-matched data, an output root, and an explicit
GPU count. The run name is appended to the output root.

```bash
uv run --no-sync python -m leanselect.ablation_campaign \
  --run minicpm_fine_s42 --model "$MINICPM_MODEL" --data "$MINICPM_TRAIN_DATA" \
  --output outputs/matched-training --gpus 8
```

Inspect the JSON `command`/`shell_command`, then run that command with the desired
`CUDA_VISIBLE_DEVICES` allocation. It launches `torch.distributed.run --standalone`
with `leanselect.train`. Change `--run` to any name below and provide the
corresponding backbone inputs. No cluster wrapper or new remote repository is
required.

| Runs | Seed(s) | Override / teacher |
|---|---|---|
| `minicpm_fine_s42`, `minicpm_fine_s43`, `minicpm_fine_s44` | 42, 43, 44 | Full selector, 2/1 |
| `minicpm_no_boundary_s42` | 42 | No boundary, 2/1 |
| `minicpm_no_worst_s42` | 42 | No worst-group, 2/1 |
| `minicpm_listwise_only_s42` | 42 | Listwise only, 2/1 |
| `minicpm_fixed_mixing_s42` | 42 | Both mixing tensors frozen, 2/1 |
| `minicpm_single_subspace_s42` | 42 | One width-128 subspace, 2/1 |
| `minicpm_coarse_s42` | 42 | Full selector, 32/16 |
| `nosa_fine_s42`, `nosa_fine_s43`, `nosa_fine_s44` | 42, 43, 44 | Full selector, 2/1 |

All runs use 300 optimizer updates, effective batch 64, per-rank batch 8,
learning rate 5e-4 with a constant schedule, weight decay 0.01, gradient norm
limit 0.5, and checkpoint export every 20 updates. MiniCPM uses 65,536 tokens;
NOSA uses 32,768. Explicit GPU counts 1, 2, 4, and 8 set accumulation to 8, 4, 2,
and 1 respectively. Each run consumes 19,200 sequence slots. GPU-count changes
preserve the effective batch but do not promise identical sampled supervision,
numerics, or resumability across world sizes.

The planner reuses the six existing override files and shared backbone configs.
For the coarse run, paired trainer flags `--training-kernel-size 32`
and `--training-kernel-stride 16` override only the teacher geometry; the resolved
configuration is recorded in the training contract and plan. Supervision remains
64 query rows with the same 96-page budget and loss coefficients. Fine training
uses the stride-1 CUDA teacher and max alignment; coarse training uses FP32
logits with the historical right-aligned causal rule. This is a **teacher-recipe
contrast**, not a pure resolution intervention; precision and causal semantics
also differ.

Seed-42 runs are current-source reference reruns. They do not replace a
previously selected best checkpoint. The public training loop exports candidate
checkpoints; it does not perform inline benchmark validation or select
a best checkpoint. Perform development-only selection separately as described in
[reproduction.md](reproduction.md). These are portable recipe commands, not a
claim of bitwise reproduction of the original orchestration. No new trained
weights or measurements are bundled.

A plain-KL objective and an unrestricted dense direct projection are not part of
these 12 runs. The one-subspace control still factorizes the head mixing and
must not be labeled an unrestricted direct projection.

## Optional short pilot

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
