# Contributing

Keep changes focused on LeanSelect training, inference, and paper evaluation.
Use a small pull request with a description of the changed behavior and tests.

```bash
uv sync --extra cpu --extra dev
uv run --extra cpu pytest
```

CUDA changes also need a real sparse-generation or training check. Report the
device, input shape, selected method, and numerical comparison. Performance
comparisons should use the same inputs, output length, device and runtime.

Preserve upstream notices. Keep credentials, datasets, model weights and local
experiment outputs outside commits.
