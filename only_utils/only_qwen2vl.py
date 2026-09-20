"""ONLY (ICCV'25) for Qwen2-VL on transformers 4.56 -- a port of the LLaVA implementation in
`transformers/src/transformers/models/llama/modeling_llama.py` + `llava_llama.py` + `only_sample.py`.

Mapping to the LLaVA reference (names from the original code):
  * layer `enhance_layer_index` ('get hidden states'): attention weights are recomputed eagerly; for the
    text columns (all non-image keys except the first sequence token, i.e. `1:35` and `35+576:` in LLaVA)
    and for the image columns, outliers > mean+std are zeroed; per-head entropy ratio text/image is taken
    on the last query row; heads with ratio < mean are removed; `hidden_states_cd = o_proj(W_cd @ V)`.
  * last layer ('last layer'): cd = input_layernorm(cd); cd = 0.2 * residual + cd; cd = cd + mlp(post_ln(cd)).
  * head: logits_cd = lm_head(norm(cd) + 0.5 * norm(h)).
  * decoding: TVD(softmax(logits), softmax(logits_cd)) < gamma -> logits + alpha_pos * logits_cd,
    else (1 + alpha_neg) * logits - alpha_neg * logits_cd; then APC cutoff log(beta) + max(logits).

Differences that are forced by the architecture, not by choice of method:
  * the image span is located from `image_token_id` (variable length) instead of the fixed [35, 611);
  * GQA: K/V are repeated to the number of query heads before the per-head statistics;
  * the first sequence token (index 0) is excluded from the text statistics exactly like LLaVA's BOS
    (for Qwen2-VL this is `<|im_start|>`); it is never removed from the sequence or from attention.
  * only the last query row of the cd branch is computed -- the original computes all rows but only the
    last position's logits are ever used, and every step of the statistics is row-wise, so the result is
    identical.

The model's own forward (main branch) is left untouched: the patched attention calls the configured
attention implementation for the main output exactly as upstream does.
"""
import math
from typing import Optional

import torch
from torch import nn
from transformers import LogitsProcessor
from transformers.models.qwen2_vl import modeling_qwen2_vl as mq


class OnlyState:
    def __init__(self):
        self.img_start = None
        self.img_end = None
        self.cd = None          # o_proj output of the cd branch at the enhance layer (last position)
        self.residual = None    # input of the last decoder layer (last position)
        self.cd_final = None    # cd after the last-layer processing
        self.h_norm = None      # final normed hidden state of the main branch (last position)
        self.tvd = []


def only_attention_cd(attn, query_states, key_states, value_states, attention_mask, img_start, img_end):
    """cd attention output for the LAST query position. Shapes: q (b, H, q, d), k/v (b, KV, k, d)."""
    bsz = query_states.shape[0]
    q = query_states[:, :, -1:, :].float()
    k = mq.repeat_kv(key_states, attn.num_key_value_groups).float()
    v = mq.repeat_kv(value_states, attn.num_key_value_groups).float()
    kv_len = k.shape[-2]

    logits = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
    if attention_mask is not None:
        m = attention_mask[:, :, -1:, :kv_len]
        if m.dtype == torch.bool:
            logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
        else:
            logits = logits + m.float()
    # (a causal mask never hides keys from the last query row, so mask=None needs no extra handling)
    w = nn.functional.softmax(logits, dim=-1)

    S, E = img_start, img_end
    w_cd = w.clone()
    zero = torch.tensor(0.0, device=w.device, dtype=w.dtype)

    text = torch.cat([w_cd[:, :, :, 1:S], w_cd[:, :, :, E:]], dim=-1)
    text = torch.where(text > text.mean(-1, keepdim=True) + text.std(-1, keepdim=True), zero, text)
    w_cd[:, :, :, 1:S] = text[:, :, :, :S - 1]
    w_cd[:, :, :, E:] = text[:, :, :, S - 1:]
    text_norm = (text / text.sum(-1, keepdim=True))[:, :, -1, :].unsqueeze(-2)

    img = w_cd[:, :, :, S:E]
    img = torch.where(img > img.mean(-1, keepdim=True) + img.std(-1, keepdim=True), zero, img)
    w_cd[:, :, :, S:E] = img
    img_norm = (img / img.sum(-1, keepdim=True))[:, :, -1, :].unsqueeze(-2)

    entropy_text = -torch.sum(text_norm * torch.log(text_norm + 1e-6), dim=-1)
    entropy_img = -torch.sum(img_norm * torch.log(img_norm + 1e-6), dim=-1)
    entropy_text = torch.nan_to_num(entropy_text, nan=0.0)
    entropy_img = torch.nan_to_num(entropy_img, nan=float('inf'))
    ratio = (entropy_text.sum(-1) / entropy_img.sum(-1)).squeeze()
    removed_heads = torch.where(ratio < ratio.mean())
    w_cd[:, removed_heads[0], :, :] = 0

    out = torch.matmul(w_cd, v).to(query_states.dtype)
    out = out.transpose(1, 2).reshape(bsz, 1, -1)
    return attn.o_proj(out)


def _make_patched_attention_forward(attn, state):
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        # --- verbatim Qwen2VLAttention.forward (transformers 4.56) ---
        bsz, q_len, _ = hidden_states.size()
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        mrope_section = getattr(attn, "rope_scaling", {}).get("mrope_section") if hasattr(attn, "rope_scaling") and isinstance(attn.rope_scaling, dict) else (
            getattr(attn.config, "rope_scaling", {}).get("mrope_section") if hasattr(attn.config, "rope_scaling") and isinstance(attn.config.rope_scaling, dict) else None
        )
        if mrope_section is not None:
            query_states, key_states = mq.apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, mrope_section
            )
        else:
            query_states, key_states = mq.apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        attention_interface = mq.eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = mq.ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            attn, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling, sliding_window=attn.sliding_window, position_ids=position_ids, **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        # --- ONLY branch ---
        state.cd = only_attention_cd(attn, query_states, key_states, value_states, attention_mask,
                                     state.img_start, state.img_end)
        return attn_output, attn_weights
    return forward


class OnlyQwen2VL:
    """Installs ONLY on a Qwen2VLForConditionalGeneration. Use `set_image_span(input_ids)` before each
    `generate` and pass `logits_processor=[OnlyLogitsProcessor(...)]` with do_sample=False."""

    def __init__(self, model, enhance_layer_index=0):
        self.model = model
        self.state = OnlyState()
        # Robust resolution of language model / text backbone across transformers versions
        lm = None
        if hasattr(model, "model") and hasattr(model.model, "language_model") and model.model.language_model is not None:
            lm = model.model.language_model
        elif hasattr(model, "language_model") and model.language_model is not None:
            lm = model.language_model
        elif hasattr(model, "get_decoder") and callable(model.get_decoder):
            lm = model.get_decoder()
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            lm = model.model
        else:
            raise AttributeError(
                f"Cannot locate language_model in {type(model).__name__}. "
                f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}"
            )

        if hasattr(lm, "layers"):
            layers = lm.layers
            self.norm = getattr(lm, "norm", None)
        elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers
            self.norm = getattr(lm.model, "norm", None)
        else:
            raise AttributeError(f"Cannot find layers in language model {type(lm).__name__}")

        if self.norm is None:
            for candidate in [lm, getattr(lm, "model", None), model, getattr(model, "model", None)]:
                if candidate is not None and hasattr(candidate, "norm"):
                    self.norm = candidate.norm
                    break

        self.lm_head = (
            getattr(model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or getattr(getattr(model, "model", None), "lm_head", None)
            or (model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None)
        )
        self.enhance_layer_index = enhance_layer_index
        self.last_index = len(layers) - 1
        self.image_token_id = getattr(model.config, "image_token_id", None)

        attn = layers[enhance_layer_index].self_attn
        self._orig_forward = attn.forward
        self._attn = attn
        attn.forward = _make_patched_attention_forward(attn, self.state)

        self._handles = []
        last = layers[self.last_index]
        state = self.state

        def pre_hook(module, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            state.residual = hs[:, -1:, :]

        def post_hook(module, args, kwargs, output):
            if enhance_layer_index == self.last_index:
                # LLaVA reference: the 'get hidden states' branch wins, no last-layer processing
                state.cd_final = state.cd
                return
            cd = state.cd.to(state.residual.device)
            cd = module.input_layernorm(cd)
            cd = 0.2 * state.residual + cd
            residual_cd = cd
            cd = module.post_attention_layernorm(cd)
            cd = module.mlp(cd)
            state.cd_final = residual_cd + cd

        def norm_hook(module, args, output):
            state.h_norm = output[:, -1:, :]

        self._handles.append(last.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(last.register_forward_hook(post_hook, with_kwargs=True))
        self._handles.append(self.norm.register_forward_hook(norm_hook))

    def set_image_span(self, input_ids):
        pos = (input_ids[0] == self.image_token_id).nonzero().flatten()
        assert pos.numel() > 0, "no image tokens in the prompt"
        start, end = pos[0].item(), pos[-1].item() + 1
        assert end - start == pos.numel(), "ONLY expects a single contiguous image span"
        assert start > 1, "text prefix before the image is required by the text statistics"
        self.state.img_start, self.state.img_end = start, end
        self.state.tvd = []

    def logits_cd(self):
        s = self.state
        device = self.lm_head.weight.device
        h_norm = s.h_norm                    # read first: calling self.norm below re-fires norm_hook
        cd_norm = self.norm(s.cd_final.to(self.norm.weight.device))
        s.h_norm = h_norm
        return self.lm_head(cd_norm.to(device) + 0.5 * h_norm.to(device))[:, -1, :]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._attn.forward = self._orig_forward


class OnlyLogitsProcessor(LogitsProcessor):
    """Must be the ONLY processor (greedy, no repetition penalty) so that `scores` are raw logits,
    matching only_sample.sample where the contrastive step precedes every processor."""

    def __init__(self, only: OnlyQwen2VL, alpha_pos=3.0, alpha_neg=1.0, beta=0.1, gamma=0.2):
        self.only, self.alpha_pos, self.alpha_neg, self.beta, self.gamma = only, alpha_pos, alpha_neg, beta, gamma

    def __call__(self, input_ids, scores):
        next_token_logits = scores
        next_token_logits_cd = self.only.logits_cd().to(scores.device, dtype=scores.dtype)
        cutoff = math.log(self.beta) + next_token_logits.max(dim=-1, keepdim=True).values
        tvd = torch.sum(torch.abs(nn.functional.softmax(next_token_logits, dim=-1)
                                  - nn.functional.softmax(next_token_logits_cd, dim=-1)))
        self.only.state.tvd.append(tvd.item())
        if tvd < self.gamma:
            diffs = next_token_logits + self.alpha_pos * next_token_logits_cd
        else:
            diffs = (1 + self.alpha_neg) * next_token_logits - self.alpha_neg * next_token_logits_cd
        return diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
