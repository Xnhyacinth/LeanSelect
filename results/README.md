# Paper results

These files contain accepted aggregate measurements from the original paper
runs, exported without raw prompts, predictions, machine paths or credentials.
They have not been recomputed in full with this extracted repository; the
release's separate correctness checks are described in
[validation](../docs/validation.md).

- `minicpm_quality.json`: five-method RULER and LongBench comparison.
- `minicpm_helmet.json`: eight methods, 13 tasks and 5,000 inputs per method.
- `minicpm_additional_ruler.json`: additional selector baselines.
- `minicpm_reasoning.json`: seven methods on MATH-500 and AIME 2024/2025.
- `nosa_ruler.csv`: five-method 32K RULER comparison, rounded as in the paper.
- `nosa_quality.json`: completed NOSA LongBench and HELMET arms.
- `*_efficiency.json`: prefill, decode and complete-request times in seconds.
- `uniform_quality.json`: paired task-stratified bootstrap differences between
  LeanSelect and Uniform, in percentage points.
- `uniform_efficiency.json`: matched native, Uniform and LeanSelect timings.

Efficiency measurements use an H200, resident KV caches, a 96-page budget and
four generated tokens. MiniCPM's eight-method comparison uses batch size one;
NOSA also includes larger batches. Ten measured requests follow warmup.
MInference, XAttention and Recency use scoring adaptations to the common page
budget, as described in the paper; these results do not measure their original
end-to-end systems. Each row identifies its length and batch size; compare
methods within the same experiment and geometry.

The result collection is still being extended. NOSA baseline LongBench/HELMET
supplements and matched reasoning runs, MiniCPM additional-baseline LongBench,
and MiniCPM InfiniGen reasoning are not yet complete. Missing arms are omitted,
not imputed. Existing completed results are retained without selecting only
favorable comparisons.
