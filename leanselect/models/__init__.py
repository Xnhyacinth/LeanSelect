"""Backbone loading and the paper's common-budget selection adapters."""

import torch

from leanselect.baselines.sparda import load_sparda_weights

METHODS = (
    "dense",
    "native",
    "sparda",
    "leanselect",
    "uniform",
    "infinigen",
    "minference",
    "xattention",
    "recency",
)


def load_model(
    backbone,
    model_path,
    method="leanselect",
    selector_checkpoint=None,
    sparda_checkpoint=None,
    device="cuda",
    dtype="bfloat16",
    sparse_config=None,
    training=False,
):
    """Load the frozen backbone and selected adapter; return ``(model, tokenizer)``.

    ``topk`` in sparse_config is the dynamic-plus-initial page budget; the
    runtime adds ``window_size // block_size`` local pages. Paper configurations
    use 96 total pages. Baseline checkpoints contain only consumed projections.
    """
    from transformers import AutoTokenizer, LlamaConfig
    from leanselect.checkpoints import (
        leanselect_checkpoint_config,
        install_leanselect_checkpoint,
    )

    if backbone not in {"minicpm", "nosa"} or method not in METHODS:
        raise ValueError(f"Unsupported backbone/method: {backbone}/{method}")
    if backbone == "nosa" and method in {
        "minference",
        "xattention",
        "recency",
        "uniform",
    }:
        raise ValueError(f"{method} is evaluated on MiniCPM only")
    if training and method != "leanselect":
        raise ValueError("Training supports the LeanSelect selector")
    if method == "leanselect" and not training and selector_checkpoint is None:
        raise ValueError("LeanSelect inference requires selector_checkpoint")
    if method == "sparda" and sparda_checkpoint is None:
        raise ValueError("SparDA requires sparda_checkpoint")
    target = torch.device(device)
    if target.type != "cuda" and method != "dense":
        raise ValueError(
            "Sparse backbones require CUDA and the native attention extension"
        )
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if target.type == "cuda":
        import importlib

        runtime = importlib.import_module(f"leanselect.models.{backbone}")
        if method != "dense":
            required = (
                "infllmv2_attn_stage1",
                "infllmv2_attn_varlen_func",
                "max_pooling_1d_varlen",
            )
            if training:
                required += ("infllmv2_attn_stage1_kl_teacher",)
            missing = [
                name for name in required if not callable(getattr(runtime, name, None))
            ]
            if missing:
                raise RuntimeError(
                    "Native sparse attention is unavailable: "
                    + ", ".join(missing)
                    + ". Build third_party/infllmv2 before loading a sparse method."
                )
        if backbone == "minicpm" and not runtime.flash_attn_available:
            raise RuntimeError("MiniCPM CUDA execution requires FlashAttention")
    model_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if model_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    checkpoint = (
        torch.load(selector_checkpoint, map_location="cpu", weights_only=True)
        if selector_checkpoint
        else None
    )
    forecast = (
        torch.load(sparda_checkpoint, map_location="cpu", weights_only=True)
        if sparda_checkpoint
        else None
    )
    if backbone == "minicpm":
        from .configuration_minicpm import MiniCPMConfig
        from .minicpm import MiniCPMForCausalLM, apply_minicpm41_128k_longrope

        config = MiniCPMConfig.from_pretrained(model_path)
        apply_minicpm41_128k_longrope(config)
        model_class = MiniCPMForCausalLM
        local_window = 2048
        if method == "infinigen":
            from .minicpm_infinigen import MiniCPMForCausalLM_InfiniGen

            model_class = MiniCPMForCausalLM_InfiniGen
    else:
        from .nosa import SparseLlamaForCausalLM

        config = LlamaConfig.from_pretrained(model_path)
        model_class = SparseLlamaForCausalLM
        local_window = 1024
    settings = dict(
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        init_blocks=1,
        window_size=local_window,
        topk=96 - local_window // 64,
        dense_len=-1,
        selection_method="current",
        create_indexer=method == "sparda",
        use_q_future_for_topk=method == "sparda",
    )
    if forecast:
        saved = forecast.get("sparse_config") or {}
        settings.update({key: value for key, value in saved.items() if key in settings})
    settings.update(sparse_config or {})
    # The public method determines the score path, independently of saved configuration.
    settings.update(
        create_indexer=method == "sparda",
        use_q_future_for_topk=method == "sparda",
        selection_method=method
        if method in {"leanselect", "uniform", "minference", "xattention", "recency"}
        else "current",
    )
    if checkpoint is not None:
        selected = leanselect_checkpoint_config(checkpoint)
        settings.update(
            leanselect_router_heads=selected["router_heads"],
            leanselect_router_dim=selected["router_dim"],
            leanselect_total_topk_blocks=selected["total_topk_blocks"],
        )
    if method == "infinigen":
        settings.update(infinigen_enabled=True, infinigen_num_channels=32)
    if method == "dense":
        settings.update(force_dense_inference=True, dense_len=2**62)
    if training:
        settings.setdefault("training_kernel_size", 2)
        settings.setdefault("training_kernel_stride", 1)
        settings.setdefault("leanselect_supervision_rows", 64)
        settings.setdefault("leanselect_worst_group_weight", 0.1)
    config.sparse_config = (
        settings if target.type == "cuda" or backbone == "nosa" else None
    )
    attention = (
        "flash_attention_2"
        if backbone == "minicpm" and target.type == "cuda"
        else "sdpa"
    )
    model = model_class.from_pretrained(
        model_path,
        config=config,
        torch_dtype=model_dtype,
        attn_implementation=attention,
    ).to(target)
    model.requires_grad_(False)
    if checkpoint is not None:
        install_leanselect_checkpoint(model, checkpoint)
    if forecast is not None:
        load_sparda_weights(model, forecast)
    for layer in model.model.layers:
        selector = getattr(layer.self_attn, "leanselect", None)
        if selector is not None:
            selector.float().requires_grad_(training)
    model.train(training)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.leanselect_method = method
    model.leanselect_backbone = backbone
    return model, tokenizer
