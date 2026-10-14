# Release validation

The current implementation passes 64 CPU tests covering the selector,
checkpoint compatibility, training, scoring, cached generation, LongBench v2
protocol/resume behavior and ablation controls. The wheel and source
distribution build successfully; the source distribution includes the ablation
configurations and benchmark documentation.

The LongBench v2 extraction was checked against the pinned upstream sources
and official aggregation code. The new benchmark and ablation entry points
have not yet been run as full GPU experiments. Use the project's own PyTorch
libraries when testing; a host `LD_LIBRARY_PATH` that places another PyTorch
installation first can load incompatible native libraries.

The initial release's fourteen CUDA checks passed on two H200 GPUs, one test process per GPU, using
real 16,384-token inputs and four generated tokens:

- MiniCPM: Dense, native sparse, SparDA, InfiniGen, Uniform, MInference,
  XAttention, Recency, and LeanSelect generation/training parity.
- NOSA: Dense, native sparse, SparDA, InfiniGen, and LeanSelect
  generation/training parity.

For both backbones, the extracted LeanSelect model produced the same tokens
as the original implementation. Last-token and full-sequence language-model
heads also produced identical generated tokens. All 32 selector layers ran;
all 128 selector parameter tensors received finite nonzero gradients and
changed after an optimizer step. Backbone parameters remained frozen.

These are correctness checks on a shared host, not a rerun of the full paper
benchmarks or an isolated performance measurement. Paper aggregates are in
[results](../results/README.md).

The pinned native CUDA source, patch application and modified-file hashes
were verified. CUDA tests used the existing native binary from the research
implementation; a fresh compilation through the public installer has not yet
been validated.
