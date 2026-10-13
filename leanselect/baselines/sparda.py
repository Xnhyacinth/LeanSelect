"""Load SparDA's consumed current-query and next-layer query projections."""

import torch


def load_sparda_weights(model, checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("SparDA checkpoint must contain a state dictionary")
    model_state = model.state_dict()
    terminal = f".layers.{len(model.model.layers) - 1}.self_attn.q_future_proj."
    expected = {
        name
        for name in model_state
        if ("q_future_proj." in name or "q_curr_proj." in name) and terminal not in name
    }
    mapped = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        prefixes = ("module.", "_orig_mod.", "base_model.model.")
        prefix = next((p for p in prefixes if name.startswith(p)), None)
        while prefix:
            name = name[len(prefix) :]
            prefix = next((p for p in prefixes if name.startswith(p)), None)
        variants = (name, name[6:] if name.startswith("model.") else "model." + name)
        target = next((v for v in variants if v in expected), None)
        if target:
            if target in mapped:
                raise ValueError(f"Duplicate SparDA tensor: {target}")
            mapped[target] = value
    if not expected or set(mapped) != expected:
        raise ValueError(f"SparDA checkpoint missing {sorted(expected - set(mapped))}")
    for name, value in mapped.items():
        if value.shape != model_state[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid SparDA tensor: {name}")
    model.load_state_dict(mapped, strict=False)
    return len(mapped)


def convert_sparda_checkpoint(source, destination, *, trust_pickle=False):
    """Export only forecast tensors from a legacy training checkpoint."""
    import hashlib
    from pathlib import Path

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location="cpu", weights_only=not trust_pickle)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("SparDA checkpoint must contain a state dictionary")
    tensors = {
        name: value.detach().cpu()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
        and ("q_future_proj." in name or "q_curr_proj." in name)
    }
    if not tensors or any(not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("Checkpoint has no finite SparDA projection tensors")
    routing = (
        "kernel_size",
        "kernel_stride",
        "block_size",
        "init_blocks",
        "window_size",
        "topk",
        "use_nope",
    )
    sparse = checkpoint.get("sparse_config") or {}
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    payload = {
        "schema": "leanselect_sparda/v1",
        "source_sha256": digest.hexdigest(),
        "model_state_dict": tensors,
        "sparse_config": {key: sparse[key] for key in routing if key in sparse},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        torch.save(payload, handle)
    return {"source_sha256": digest.hexdigest(), "tensors": len(tensors)}


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Export tensor-only SparDA baseline weights"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Explicitly allow unpickling a trusted legacy training checkpoint",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            convert_sparda_checkpoint(
                args.source, args.output, trust_pickle=args.trust_pickle
            )
        )
    )


if __name__ == "__main__":
    main()
