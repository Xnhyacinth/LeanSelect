"""Canonical selector checkpoints and an explicit reader for legacy artifacts."""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

from .selector import LEANSELECT_CANONICAL_TENSORS

LEANSELECT_SCHEMA = "leanselect_selector/v1"
LEANSELECT_PROVENANCE_SCHEMA = "leanselect_training_provenance/v1"
_LEANSELECT_CONFIG_KEYS = {"router_heads", "router_dim", "num_heads", "num_kv_heads", "head_dim", "plan_scope", "page_aggregation", "score_contract", "total_topk_blocks", "forced_init_blocks", "forced_local_blocks"}
_LEANSELECT_PROVENANCE_KEYS = {"schema", "base_model_identity", "base_model_config_sha256", "training_step", "run_contract_sha256", "data_manifest_sha256"}


def normalize_checkpoint(checkpoint: Mapping) -> dict:
    """Translate the legacy envelope without mutating the input or its tensors."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if "leanselect" in checkpoint and "phase_tiny" in checkpoint:
        raise ValueError("checkpoint contains ambiguous selector payloads")
    if "leanselect" in checkpoint:
        return dict(checkpoint)
    legacy = checkpoint.get("phase_tiny")
    if not isinstance(legacy, Mapping) or legacy.get("schema") != "phase_tiny_selector/v3":
        raise ValueError("checkpoint must contain a supported selector payload")
    payload = dict(legacy)
    payload["schema"] = LEANSELECT_SCHEMA
    if "provenance" in payload:
        provenance = payload["provenance"]
        if not isinstance(provenance, Mapping) or provenance.get("schema") != "phase_tiny_training_provenance/v1":
            raise ValueError("legacy training provenance schema is incompatible")
        payload["provenance"] = {**provenance, "schema": LEANSELECT_PROVENANCE_SCHEMA}
    return {"leanselect": payload}


@lru_cache(maxsize=16)
def _cached_base_model_weights_manifest_sha256(
    root_text: str, fingerprint: tuple[tuple[str, int, int], ...]
) -> str:
    root = Path(root_text)
    entries = []
    for relative, size, _mtime_ns in fingerprint:
        path = root / relative
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"path": relative, "size": size, "sha256": digest.hexdigest()})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def base_model_weights_manifest_sha256(model_path: str | Path) -> str:
    """Hash every local model weight shard and its exact manifest."""

    root = Path(model_path).resolve()
    files = sorted(
        {
            *root.glob("*.safetensors"),
            *root.glob("pytorch_model*.bin"),
        },
        key=lambda path: path.name,
    )
    if not files:
        raise ValueError(f"base model has no local weight shards: {root}")
    fingerprint = tuple(
        (path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    )
    return _cached_base_model_weights_manifest_sha256(str(root), fingerprint)


def _leanselect_config(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    router_heads: int,
    router_dim: int,
    total_topk_blocks: int,
    forced_init_blocks: int,
    forced_local_blocks: int,
) -> dict[str, object]:
    return {
        "router_heads": int(router_heads),
        "router_dim": int(router_dim),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "plan_scope": "per_gqa",
        "page_aggregation": "overlapping_atom_max",
        "score_contract": "per_gqa_projected_atom_dot",
        "total_topk_blocks": int(total_topk_blocks),
        "forced_init_blocks": int(forced_init_blocks),
        "forced_local_blocks": int(forced_local_blocks),
    }


def build_leanselect_checkpoint(
    layers: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    provenance: Mapping[str, object] | None = None,
    **config,
) -> dict[str, object]:
    """Build a portable checkpoint from canonical FP32 selector tensors."""

    payload_layers = {}
    for layer, state in sorted(layers.items()):
        if set(state) != set(LEANSELECT_CANONICAL_TENSORS):
            raise ValueError(
                "LeanSelect layer must contain exactly the canonical tensors"
            )
        payload_layers[str(int(layer))] = {
            name: state[name].detach().to(device="cpu", dtype=torch.float32).clone()
            for name in LEANSELECT_CANONICAL_TENSORS
        }
    payload = {
        "schema": LEANSELECT_SCHEMA,
        "role": "trained",
        "config": _leanselect_config(**config),
        "layers": payload_layers,
    }
    if provenance is not None:
        payload["provenance"] = _validate_leanselect_provenance(provenance)
    return {"leanselect": payload}


def _validate_leanselect_provenance(provenance: Mapping) -> dict[str, object]:
    if not isinstance(provenance, Mapping) or set(provenance) != (
        _LEANSELECT_PROVENANCE_KEYS
    ):
        raise ValueError("LeanSelect training provenance keys are incompatible")
    checked = dict(provenance)
    if checked["schema"] != LEANSELECT_PROVENANCE_SCHEMA:
        raise ValueError("LeanSelect training provenance schema is incompatible")
    if (
        not isinstance(checked["base_model_identity"], str)
        or not checked["base_model_identity"]
    ):
        raise ValueError("LeanSelect training provenance base model is missing")
    if not isinstance(checked["training_step"], int) or checked["training_step"] < 0:
        raise ValueError("LeanSelect training provenance step is invalid")
    for name in (
        "base_model_config_sha256",
        "run_contract_sha256",
        "data_manifest_sha256",
    ):
        value = checked[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"LeanSelect training provenance {name} is invalid")
    return checked


def leanselect_checkpoint_provenance(
    checkpoint: Mapping, *, required: bool = False
) -> dict[str, object] | None:
    """Return validated training provenance.

    Use ``required=True`` when loading an artifact that must identify its
    training run.
    """

    checkpoint = normalize_checkpoint(checkpoint)
    payload = checkpoint.get("leanselect")
    if not isinstance(payload, Mapping) or payload.get("schema") != LEANSELECT_SCHEMA:
        raise ValueError(f"checkpoint must contain {LEANSELECT_SCHEMA}")
    provenance = payload.get("provenance")
    if provenance is None:
        if required:
            raise ValueError("LeanSelect checkpoint is missing training provenance")
        return None
    return _validate_leanselect_provenance(provenance)


def leanselect_checkpoint_config(checkpoint: Mapping) -> dict[str, object]:
    checkpoint = normalize_checkpoint(checkpoint)
    payload = checkpoint.get("leanselect")
    if not isinstance(payload, Mapping) or payload.get("schema") != LEANSELECT_SCHEMA:
        raise ValueError(f"checkpoint must contain {LEANSELECT_SCHEMA}")
    if payload.get("role") != "trained":
        raise ValueError("LeanSelect deployment checkpoint role must be trained")
    leanselect_checkpoint_provenance(checkpoint)
    config = payload.get("config")
    if not isinstance(config, Mapping) or set(config) != _LEANSELECT_CONFIG_KEYS:
        raise ValueError("LeanSelect checkpoint config keys are incompatible")
    return dict(config)


def validate_leanselect_checkpoint(
    checkpoint: Mapping,
    *,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    total_topk_blocks: int,
    forced_init_blocks: int,
    forced_local_blocks: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Validate model geometry and return canonical per-layer FP32 tensors."""

    checkpoint = normalize_checkpoint(checkpoint)
    config = leanselect_checkpoint_config(checkpoint)
    expected_runtime = {
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "plan_scope": "per_gqa",
        "page_aggregation": "overlapping_atom_max",
        "score_contract": "per_gqa_projected_atom_dot",
        "total_topk_blocks": int(total_topk_blocks),
        "forced_init_blocks": int(forced_init_blocks),
        "forced_local_blocks": int(forced_local_blocks),
    }
    mismatches = {
        name: (value, config.get(name))
        for name, value in expected_runtime.items()
        if config.get(name) != value
    }
    if mismatches:
        raise ValueError(f"LeanSelect checkpoint runtime config mismatch: {mismatches}")
    hr, dr = int(config["router_heads"]), int(config["router_dim"])
    if (
        min(int(num_layers), int(num_heads), int(num_kv_heads), int(head_dim), hr, dr)
        <= 0
        or int(num_heads) % int(num_kv_heads)
        or hr > int(num_heads)
    ):
        raise ValueError("LeanSelect checkpoint model/router geometry is invalid")
    expected_shapes = {
        "head_mix": (int(num_heads), hr),
        "kv_head_mix": (int(num_kv_heads), hr),
        "q_proj": (hr, int(head_dim), dr),
        "k_proj": (hr, int(head_dim), dr),
    }
    payload_layers = checkpoint["leanselect"].get("layers")
    expected_ids = {str(index) for index in range(int(num_layers))}
    if not isinstance(payload_layers, Mapping) or set(payload_layers) != expected_ids:
        raise ValueError("LeanSelect checkpoint layers must exactly cover the model")
    result = {}
    for index in range(int(num_layers)):
        state = payload_layers[str(index)]
        if not isinstance(state, Mapping) or set(state) != set(
            LEANSELECT_CANONICAL_TENSORS
        ):
            raise ValueError("LeanSelect layer has incompatible canonical tensors")
        checked = {}
        for name, shape in expected_shapes.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"LeanSelect layer {index} tensor {name} shape mismatch"
                )
            if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                raise ValueError(
                    "LeanSelect checkpoint canonical tensors must be CPU FP32"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    "LeanSelect checkpoint canonical tensors must be finite"
                )
            checked[name] = tensor
        result[index] = checked
    return result


def install_leanselect_checkpoint(model, checkpoint: Mapping) -> int:
    """Copy validated canonical tensors onto a constructed MiniCPM LeanSelect model."""

    layers = model.model.layers
    if not layers:
        raise ValueError("MiniCPM model has no layers")
    first = layers[0].self_attn
    if getattr(first, "leanselect", None) is None:
        raise ValueError(
            "LeanSelect checkpoint requires selection_method=leanselect before "
            "model construction"
        )
    canonical = validate_leanselect_checkpoint(
        checkpoint,
        num_layers=len(layers),
        num_heads=int(first.num_heads),
        num_kv_heads=int(first.num_key_value_heads),
        head_dim=int(first.head_dim),
        total_topk_blocks=int(first.leanselect_total_topk_blocks),
        forced_init_blocks=int(first.init_blocks),
        forced_local_blocks=int(first.local_blocks),
    )
    for index, layer in enumerate(layers):
        selector = layer.self_attn.leanselect
        if selector is None:
            raise ValueError(f"layer {index} is missing LeanSelectSelector")
        for name, tensor in canonical[index].items():
            param = getattr(selector, name)
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
    return len(layers)
