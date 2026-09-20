import os
import sys

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import pytest
except ImportError:
    pytest = None
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration, LogitsProcessorList

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from only_utils.only_llava import OnlyLlava, OnlyLlavaLogitsProcessor  # noqa: E402

IMAGE_TOKEN = 999


def tiny_llava_model():
    torch.manual_seed(0)
    vision_config = CLIPVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=56,
        patch_size=14,
    )
    text_config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    cfg = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        ignore_index=-100,
        image_token_index=IMAGE_TOKEN,
    )
    model = LlavaForConditionalGeneration(cfg).eval()
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    return model


def tiny_llava_inputs(model):
    torch.manual_seed(1)
    pixel_values = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        feats = model.get_image_features(pixel_values)
    feat_tensor = feats[0] if isinstance(feats, (list, tuple)) else feats
    num_patches = feat_tensor.shape[0] if feat_tensor.ndim == 2 else feat_tensor.shape[1]
    prefix = [1] + torch.randint(10, 500, (4,)).tolist()  # BOS + 4 text tokens
    suffix = torch.randint(10, 500, (5,)).tolist()         # 5 text tokens
    ids = prefix + [IMAGE_TOKEN] * num_patches + suffix
    input_ids = torch.tensor([ids], dtype=torch.long)
    return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=pixel_values), num_patches


def test_only_llava_hooks_and_generation():
    model = tiny_llava_model()
    inputs, num_patches = tiny_llava_inputs(model)

    # Base greedy output without ONLY
    out_base = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert out_base is not None

    # Install ONLY
    only = OnlyLlava(model, enhance_layer_index=0)
    only.set_image_span(inputs["input_ids"], image_token_id=IMAGE_TOKEN)

    # Image span should be start=5 (1 BOS + 4 text), end=5+num_patches
    assert only.state.img_start == 5
    assert only.state.img_end == 5 + num_patches

    # Generate with ONLY processor
    proc = OnlyLlavaLogitsProcessor(only, alpha_pos=3.0, alpha_neg=1.0, beta=0.1, gamma=0.2)
    out_only = model.generate(**inputs, max_new_tokens=4, do_sample=False, logits_processor=LogitsProcessorList([proc]))

    assert out_only is not None
    assert len(only.state.tvd) == 4
    for tvd_val in only.state.tvd:
        assert not torch.isnan(torch.tensor(tvd_val))

    # Remove hooks
    only.remove()
    print("[PASSED] test_only_llava_hooks_and_generation PASSED!")


if __name__ == "__main__":
    test_only_llava_hooks_and_generation()
