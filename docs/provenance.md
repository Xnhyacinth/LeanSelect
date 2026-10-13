# Source provenance

LeanSelect extracts the Paper1 selector and its training and evaluation paths
from [SparDA](https://github.com/Xnhyacinth/sparda). It includes MiniCPM and
NOSA adapters and the baseline scoring implementations needed for matched
page-budget comparisons. Unrelated architecture experiments, run orchestration,
training data and model checkpoints are excluded.

| Component | Origin |
| --- | --- |
| Selector, supervision, objective | Paper1 implementation in SparDA |
| MiniCPM adapter | OpenBMB MiniCPM and SparDA |
| NOSA adapter | THUNLP NOSA, Hugging Face Llama and SparDA |
| Baseline scoring | SparDA, InfiniGen and the paper's page-scoring adaptations |
| Native sparse CUDA kernels | OpenBMB InfLLM V2, FlashAttention and CUTLASS |
| Benchmark protocols | Official LongBench, RULER, HELMET and NOSA evaluation code |

Exact kernel revisions, patch and source hashes are recorded in
[third_party/infllmv2](../third_party/infllmv2/README.md). Benchmark revisions
are pinned by the evaluation entry point. Original copyright and license
notices are retained; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
