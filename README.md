# LeanSelect

**LeanSelect: Efficient Long-Context Prefill with Lightweight Selection**

LeanSelect reduces the cost of selecting sparse-attention pages during long-context prefill. It reuses backbone queries and compressed keys, learns page scores with budget-aligned supervision, and folds head-mixing coefficients into the inference projections. The backbone remains frozen.

This repository contains the selector, MiniCPM4.1 and NOSA integrations, training, and quality and efficiency evaluation. Supply the pretrained backbone, training and benchmark data, and any baseline checkpoints separately.

## Install

Use Linux, Python 3.10, and a CUDA 12.8 toolkit for GPU execution:

```bash
uv sync --extra cuda --extra eval --extra dev
uv pip install ninja packaging psutil
uv pip install --no-build-isolation flash-attn==2.8.3
uv run --no-sync python third_party/infllmv2/install.py --directory .build/infllmv2
```

The [native extension installer](third_party/infllmv2/README.md) downloads pinned InfLLM V2/CUTLASS sources and applies the included patch. Use `uv run --no-sync` after installing the extensions so subsequent commands retain these separately built packages.

For CPU development:

```bash
uv sync --extra cpu --extra dev
uv run --no-sync pytest
```

## Reproduce

See [reproduction.md](docs/reproduction.md) for four-GPU training, development-set checkpoint selection, benchmark preparation, and timing commands.

[Results](results/README.md) lists the exported paper results and their coverage. [Validation](docs/validation.md) records the checks performed on this implementation.

| Method (`--method`) | MiniCPM4.1 | NOSA | Additional weights |
|---|:---:|:---:|---|
| LeanSelect (`leanselect`) | ✓ | ✓ | Selected selector checkpoint |
| Dense (`dense`) / native sparse (`native`) | ✓ | ✓ | — |
| SparDA (`sparda`) | ✓ | ✓ | SparDA projections |
| InfiniGen (`infinigen`) | ✓ | ✓ | — |
| Uniform (`uniform`) | ✓ | — | — |
| MInference (`minference`) | ✓ | — | — |
| XAttention (`xattention`) | ✓ | — | — |
| Recency (`recency`) | ✓ | — | — |

The selection baselines use the paper's common-page-budget adapters. Quality evaluation retains each benchmark's prompts and metrics; efficiency evaluation records warmed prefill, decode, and complete-generation times on fixed token inputs.

## Layout

- `leanselect/`: selector, objective, backbones, baselines, and command-line tools.
- `configs/`: the two backbone configurations.
- `third_party/infllmv2/`: pinned native extension installer and patch.
- `tests/`: selector, runtime, training-resume, and evaluation checks.

## License

LeanSelect is distributed under Apache-2.0. Adapted components retain their original licenses and attribution; see [NOTICE](NOTICE) and [third_party/infllmv2/README.md](third_party/infllmv2/README.md).
