"""CPU tests for only_utils/only_qwen2vl.py on a tiny random Qwen2-VL (transformers 4.56).

python -m pytest tests/test_only_qwen2vl.py -q
"""
import os
import sys

import pytest
import torch
from torch import nn
from transformers import LogitsProcessorList, Qwen2VLConfig, Qwen2VLForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from only_utils.only_qwen2vl import OnlyLogitsProcessor, OnlyQwen2VL, only_attention_cd  # noqa: E402

IMAGE_TOKEN = 151655
VISION_START = 151652
VISION_END = 151653


def tiny_model(attn_impl):
    torch.manual_seed(0)
    cfg = Qwen2VLConfig(
        vision_config=dict(depth=1, embed_dim=32, num_heads=2, hidden_size=64, mlp_ratio=2,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_chans=3),
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}, rope_theta=10000.0),
        image_token_id=IMAGE_TOKEN, vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
        attn_implementation=attn_impl,
    )
    model = Qwen2VLForConditionalGeneration(cfg).eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


def tiny_inputs():
    torch.manual_seed(1)
    grid = torch.tensor([[1, 8, 8]])                     # 64 patches -> 16 merged image tokens
    pixel_values = torch.randn(64, 3 * 2 * 14 * 14)
    prefix = torch.randint(10, 1000, (7,)).tolist()
    suffix = torch.randint(10, 1000, (9,)).tolist()
    ids = prefix + [VISION_START] + [IMAGE_TOKEN] * 16 + [VISION_END] + suffix
    input_ids = torch.tensor([ids])
    return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values, image_grid_thw=grid)


def greedy(model, inputs, processors=None, n=12):
    out = model.generate(**inputs, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None,
                         repetition_penalty=1.0, logits_processor=processors)
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def reference_cd_all_rows(attn, q, k, v, mask, S, E):
    """Literal port of the LLaVA code (all query rows, 35 -> S, 35+576 -> E)."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv
    k = repeat_kv(k, attn.num_key_value_groups).float()
    v = repeat_kv(v, attn.num_key_value_groups).float()
    q = q.float()
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn.scaling + mask
    w = nn.functional.softmax(attn_weights, dim=-1)
    w_cd = w.clone()
    t = torch.cat([w_cd[:, :, :, 1:S], w_cd[:, :, :, E:]], dim=-1)
    t = torch.where(t > t.mean(-1, keepdim=True) + t.std(-1, keepdim=True), torch.tensor(0.0), t)
    w_cd[:, :, :, 1:S] = t[:, :, :, :S - 1]
    w_cd[:, :, :, E:] = t[:, :, :, S - 1:]
    tn = (t / t.sum(-1, keepdim=True))[:, :, -1, :].unsqueeze(-2)
    im = w_cd[:, :, :, S:E]
    im = torch.where(im > im.mean(-1, keepdim=True) + im.std(-1, keepdim=True), torch.tensor(0.0), im)
    w_cd[:, :, :, S:E] = im
    imn = (im / im.sum(-1, keepdim=True))[:, :, -1, :].unsqueeze(-2)
    et = torch.nan_to_num(-torch.sum(tn * torch.log(tn + 1e-6), dim=-1), nan=0.0)
    ei = torch.nan_to_num(-torch.sum(imn * torch.log(imn + 1e-6), dim=-1), nan=float('inf'))
    ratio = (et.sum(-1) / ei.sum(-1)).squeeze()
    w_cd[:, torch.where(ratio < ratio.mean())[0], :, :] = 0
    out = torch.matmul(w_cd, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    return attn.o_proj(out)


def test_cd_last_row_matches_full_reference():
    model = tiny_model("eager")
    attn = model.model.language_model.layers[0].self_attn
    torch.manual_seed(3)
    L, S, E = 40, 9, 25
    q = torch.randn(1, 4, L, 16)
    k = torch.randn(1, 2, L, 16)
    v = torch.randn(1, 2, L, 16)
    mask = torch.full((L, L), torch.finfo(torch.float32).min).triu(1)[None, None]
    ref = reference_cd_all_rows(attn, q, k, v, mask, S, E)[:, -1:, :]
    for m in (mask, mask == 0, None):   # additive (eager), boolean (sdpa), skipped mask
        got = only_attention_cd(attn, q, k, v, m, S, E)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_patch_does_not_change_main_branch(attn_impl):
    model = tiny_model(attn_impl)
    inputs = tiny_inputs()
    base = greedy(model, inputs)
    only = OnlyQwen2VL(model, enhance_layer_index=0)
    only.set_image_span(inputs["input_ids"])
    assert (only.state.img_start, only.state.img_end) == (8, 24)
    assert greedy(model, inputs) == base          # hooks installed, no processor
    only.remove()
    assert greedy(model, inputs) == base


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_degenerate_only_equals_greedy(attn_impl):
    """alpha_pos=0 with gamma=inf -> diffs == logits; cutoff never removes the argmax."""
    model = tiny_model(attn_impl)
    inputs = tiny_inputs()
    base = greedy(model, inputs)
    only = OnlyQwen2VL(model, enhance_layer_index=0)
    only.set_image_span(inputs["input_ids"])
    proc = OnlyLogitsProcessor(only, alpha_pos=0.0, alpha_neg=1.0, beta=0.1, gamma=float("inf"))
    assert greedy(model, inputs, LogitsProcessorList([proc])) == base
    assert len(only.state.tvd) == 12
    only.remove()


def test_logits_cd_matches_manual_composition():
    """Re-derive logits_cd for the prefill step from hidden states and compare with the hooks."""
    model = tiny_model("eager")
    inputs = tiny_inputs()
    only = OnlyQwen2VL(model, enhance_layer_index=0)
    only.set_image_span(inputs["input_ids"])
    captured = {}
    lm = model.model.language_model
    layer0 = lm.layers[0]

    def cap_attn_in(module, args, kwargs):
        captured["attn_in"] = kwargs.get("hidden_states", args[0] if args else None)
    h = layer0.self_attn.register_forward_pre_hook(cap_attn_in, with_kwargs=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h.remove()
    got = only.logits_cd()

    hs = out.hidden_states                       # (emb, after L0, after L1, final normed)
    last = lm.layers[-1]
    cd = only.state.cd                           # validated against the reference in the test above
    cd = last.input_layernorm(cd)
    cd = 0.2 * hs[-2][:, -1:, :] + cd            # input to the last layer
    cd = cd + last.mlp(last.post_attention_layernorm(cd))
    want = model.lm_head(lm.norm(cd) + 0.5 * hs[-1][:, -1:, :])[:, -1, :]
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(out.logits[:, -1, :], model.lm_head(hs[-1][:, -1:, :])[:, -1, :])
    only.remove()
