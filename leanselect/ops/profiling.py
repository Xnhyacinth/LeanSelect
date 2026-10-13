"""NVTX annotations on CUDA builds; no annotations on CPU-only PyTorch."""

import torch


def _noop(*args):
    return None


if torch.version.cuda is None:
    range_push = _noop
    range_pop = _noop
else:
    range_push = torch.cuda.nvtx.range_push
    range_pop = torch.cuda.nvtx.range_pop
