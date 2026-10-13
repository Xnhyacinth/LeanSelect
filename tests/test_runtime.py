"""Small CPU models exercise the actual backbone forward and generation code."""

import pytest
import torch
from transformers import LlamaConfig

from leanselect.models import load_model
from leanselect.models.minicpm import (
    MiniCPMConfig,
    MiniCPMForCausalLM,
    MiniCPMInfLLMv2Attention,
)
from leanselect.models.nosa import SparseLlamaForCausalLM
from leanselect.baselines.sparda import load_sparda_weights


def tiny_config(backbone):
    fields = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    if backbone == "minicpm":
        config = MiniCPMConfig(**fields, scale_emb=1, dim_model_base=64, scale_depth=1)
        config.sparse_config = None
    else:
        config = LlamaConfig(**fields)
        config.sparse_config = dict(
            force_dense_inference=True,
            use_q_future_for_topk=False,
            create_indexer=False,
        )
    config._attn_implementation = "sdpa"
    return config


@pytest.mark.parametrize("backbone", ["minicpm", "nosa"])
def test_dense_cached_generation_matches_full_prefix(backbone):
    torch.manual_seed(7)
    cls = MiniCPMForCausalLM if backbone == "minicpm" else SparseLlamaForCausalLM
    model = cls(tiny_config(backbone)).eval()
    inputs = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
        )
        reference = inputs
        for _ in range(3):
            output = model(
                reference, attention_mask=torch.ones_like(reference), use_cache=False
            )
            reference = torch.cat(
                (reference, output.logits[:, -1].argmax(-1, keepdim=True)), -1
            )
    assert torch.equal(generated, reference)


def test_leanselect_and_uniform_routing_respect_visibility():
    config = tiny_config("minicpm")
    config._attn_implementation = "flash_attention_2"
    config.sparse_config = dict(
        selection_method="leanselect",
        create_indexer=False,
        topk=2,
        window_size=64,
        leanselect_router_heads=2,
        leanselect_router_dim=8,
    )
    attn = MiniCPMInfLLMv2Attention(config, 0)
    queries, keys = torch.randn(256, 4, 16), torch.randn(15, 2, 16)
    q_offsets = torch.tensor([0, 256], dtype=torch.int32)
    k_offsets = torch.tensor([0, 15], dtype=torch.int32)
    for method in ("leanselect", "uniform"):
        attn.selection_method = method
        selected = attn._lightweight_compressed_attention(
            queries, keys, q_offsets, k_offsets, 256
        )
        visible = torch.arange(256)[None, :, None] // 64
        assert selected.dtype == torch.int32
        assert torch.all((selected < 0) | (selected <= visible))
        assert torch.all((selected == 0).any(-1))


def test_sparda_checkpoint_requires_all_consumed_layers():
    config = tiny_config("minicpm")
    config._attn_implementation = "flash_attention_2"
    config.sparse_config = dict(create_indexer=True, use_q_future_for_topk=True)
    attns = [MiniCPMInfLLMv2Attention(config, i) for i in range(2)]

    class Layer(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.self_attn = attn

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([Layer(a) for a in attns])
    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if "q_curr_proj" in name or "q_future_proj" in name
    }
    state.pop(
        "model.layers.1.self_attn.q_future_proj.weight"
    )  # No downstream consumer.
    assert load_sparda_weights(model, {"model_state_dict": state}) == 2
    state.pop("model.layers.0.self_attn.q_future_proj.weight")
    with pytest.raises(ValueError, match="missing"):
        load_sparda_weights(model, {"model_state_dict": state})


def test_sparse_loader_rejects_cpu_before_model_access():
    with pytest.raises(ValueError, match="require CUDA"):
        load_model("minicpm", "nonexistent", method="native", device="cpu")


def test_missing_native_kernel_rejected_before_weights(monkeypatch):
    import leanselect.models.minicpm as runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "infllmv2_attn_stage1", None)
    with pytest.raises(RuntimeError, match="Native sparse attention is unavailable"):
        load_model("minicpm", "nonexistent", method="native", device="cuda")


@pytest.mark.parametrize("backbone", ["minicpm", "nosa"])
def test_infinigen_calibration_is_stable_and_preserves_generation_rng(backbone):
    from types import SimpleNamespace
    from leanselect.baselines.infinigen import calibrate_infinigen

    class CalibrationRecorder:
        device = torch.device("cpu")
        config = SimpleNamespace(vocab_size=101)
        leanselect_backbone = backbone

        def __init__(self):
            self.attn = SimpleNamespace(_ig_warmed_up=False)
            self.inputs = []

        def _get_infinigen_attns(self):
            return [self.attn]

        def infinigen_warmup(self, input_ids, max_new_tokens):
            self.inputs.append(input_ids.clone())
            assert not self.attn._ig_warmed_up
            torch.rand(
                3
            )  # Even incidental warmup randomness must not affect generation.
            self.attn._ig_warmed_up = True

        def _cleanup_infinigen_request_state(self):
            pass

    model = CalibrationRecorder()
    prompt = torch.arange(3000).view(1, -1)
    before = torch.random.get_rng_state().clone()
    first = calibrate_infinigen(model, prompt, 4)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(19)
    before = torch.random.get_rng_state().clone()
    second = calibrate_infinigen(model, prompt, 4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first == second
    assert torch.equal(model.inputs[0], model.inputs[1])
    assert model.inputs[0].shape == (1, 2048)
    if backbone == "minicpm":
        assert torch.equal(model.inputs[0], prompt[:, :2048])
    else:
        expected = torch.randint(
            0, 101, (1, 2048), generator=torch.Generator().manual_seed(42)
        )
        assert torch.equal(model.inputs[0], expected)


def test_sparse_decoder_constructs_only_its_active_selector(monkeypatch):
    from leanselect.models.minicpm import MiniCPMDecoderLayer

    config = tiny_config("minicpm")
    config._attn_implementation = "flash_attention_2"
    config.sparse_config = dict(
        selection_method="leanselect",
        create_indexer=False,
        leanselect_router_heads=2,
        leanselect_router_dim=8,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    layer = MiniCPMDecoderLayer(config, 0)
    assert layer.use_sparse_attn
    assert {id(p) for p in layer.indexer_curr_params} == {
        id(p) for p in layer.self_attn.leanselect.parameters()
    }
    assert layer.indexer_future_params == []


def test_sparse_decode_rope_includes_cached_prefix(monkeypatch):
    from transformers.cache_utils import DynamicCache

    config = tiny_config("minicpm")
    config._attn_implementation = "flash_attention_2"
    config.sparse_config = dict(
        selection_method="current",
        create_indexer=False,
        use_q_future_for_topk=False,
        dense_len=-1,
    )
    attn = MiniCPMInfLLMv2Attention(config, 0).eval()
    cache = DynamicCache()
    keys = torch.randn(1, 2, 3, 16)
    cache.update(keys, keys.clone(), 0)
    # Isolate rotary/cache handling from the separately tested CUDA attention op.
    monkeypatch.setattr(
        attn,
        "_sparse_attention_forward",
        lambda q, *a, **kw: (torch.zeros_like(q), (None, None)),
    )
    with torch.no_grad():
        output = attn(
            torch.randn(1, 1, 64),
            attention_mask=torch.ones(1, 4),
            position_ids=torch.tensor([[3]]),
            past_key_value=cache,
            use_cache=True,
        )
    assert output[0].shape == (1, 1, 64)
    assert cache.get_seq_length() == 4
