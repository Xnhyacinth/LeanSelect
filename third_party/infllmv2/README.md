# Native sparse attention

The runtime uses OpenBMB's InfLLM V2 attention kernels. `install.py` fetches a
pinned public revision, applies `paper1.patch`, verifies the modified source
hashes, and builds it in the active Python environment. CUTLASS is pinned
separately. CUDA sources and vendor headers are downloaded during installation.

```bash
uv run python third_party/infllmv2/install.py --directory .build/infllmv2
```

Install PyTorch, a compatible CUDA toolkit, `ninja`, `packaging`, `psutil`, and
FlashAttention first. The patch includes the sampled teacher kernel, aligned
causal masking, sparse page pooling, and the compatibility fixes used by the
paper's model adapters. `--prepare-only` downloads and verifies source without
compiling. Model weights and datasets are not included.

Original authorship and per-file licenses remain in the patch: OpenBMB
(Apache-2.0), FlashAttention / Tri Dao (BSD-3-Clause), and the documented
Apache-2.0 modifications. CUTLASS retains its upstream NVIDIA license.
