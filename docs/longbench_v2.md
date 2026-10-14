# LongBench v2

`python -m leanselect.longbench_v2` runs **direct 0-shot answer generation** with local Transformers models. It uses the official prompt and exact answer extractor from the pinned LongBench checkout. It is a separate protocol from LongBench v1, answer-choice likelihood evaluation, CoT, and RAG. Do not pool their scores.

Prepare the public benchmark independently:

```bash
git clone https://github.com/THUDM/LongBench third_party/LongBench
git -C third_party/LongBench checkout 2e00731f8d0bff23dc4325161044d0ed8af94c1e
hf download zai-org/LongBench-v2 data.json --repo-type dataset \
  --revision 2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9 \
  --local-dir datasets/longbench-v2
```

The runner requires the exact official `data.json` SHA256 `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`, all 503 unique IDs, and unmodified upstream sources. `--subset` explicitly enables a JSON array or JSONL subset with the same fields for development and smoke checks; these outputs are labeled partial/custom inputs. Dataset text is not bundled with this repository.

After the normal installation and checkpoint preparation in the project README:

```bash
uv run --no-sync python -m leanselect.longbench_v2 \
  --backbone minicpm --model "$MODEL_DIR" --method leanselect \
  --selector-checkpoint "$SELECTOR_CHECKPOINT" \
  --upstream third_party/LongBench --inputs datasets/longbench-v2/data.json \
  --output outputs/lbv2/leanselect
```

Use a separate output directory for each method. `--method dense`, `native`, `sparda` (with `--sparda-checkpoint`), and the other methods in the common runtime are available. MInference and XAttention here are local scoring adapters, not complete upstream systems. `--resume` requires an unchanged model/source/data/configuration contract, exact ordered IDs, metadata, extracted scores, seeds, and actual tokenized prompts. It calibrates InfiniGen again on the original first example before resuming.

The default protocol is:

- Encode the official raw prompt with the tokenizer's default special-token behavior; retain its first/last 60,000 tokens when over the 120,000-token budget, decode with `skip_special_tokens=True`, then apply one native user chat template.
- Tokenize the resulting chat without adding special tokens again. Reject inputs whose length plus 128 output tokens exceeds the explicit 131,072 context cap. `--prompt-budget` and `--context-cap` record any requested adaptation; choose settings supported by the actual backbone.
- Sample at temperature 0.1, top-p 1, top-k 0, maximum 128 new tokens, repetition penalty 1, one beam, natural native EOS. Seed each request with `42 + input row index` by default.
- Keep the tokenizer's native default thinking policy. This does not introduce an explicit CoT instruction or a second answer-extraction generation pass. The backend is a Transformers adaptation of the official API protocol, with no claim of identical sampled outputs.
- Apply the pinned official case-sensitive extractor; bare letters, lowercase alternatives, and unparsable responses receive no relaxed fallback or chance compensation. Report official one-decimal Overall/Easy/Hard/Short/Medium/Long accuracies and denominators. Empty subset groups have null accuracy.

`run.json`, `predictions.jsonl`, and `summary.json` contain local reproducibility information and model responses. Keep these in the ignored `outputs/` directory and review access before sharing. Predictions store IDs, evaluation metadata, responses, scores, seeds, token counts, and token hashes; they do not copy source contexts or questions. CPU regression tests cover prompt construction, truncation, extraction, aggregation, context rejection, dataset validation, and interrupted-run continuation. Full model/GPU evaluation remains a separate runtime check.
