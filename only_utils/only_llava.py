"""ONLY (ICCV'25) for Llava-1.5 on transformers >= 4.45.0 (LlavaForConditionalGeneration).

Clean, modular implementation using PyTorch forward hooks and LogitsProcessor.
Works seamlessly with HuggingFace Hub checkpoints (`llava-hf/llava-1.5-7b-hf`),
BF16 Full Precision, and multi-GPU `device_map="auto"`.
"""
import math
from typing import Optional

import torch
from torch import nn
from transformers import LogitsProcessor


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent to transformers.models.llama.modeling_llama.repeat_kv"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class OnlyLlavaState:
    def __init__(self):
        self.img_start = 35
        self.img_end = 35 + 576  # 611
        self.cd = None          # o_proj output of the cd branch at the enhance layer (last position)
        self.residual = None    # input of the last decoder layer (last position)
        self.cd_final = None    # cd after the last-layer processing
        self.h_norm = None      # final normed hidden state of the main branch (last position)
        self.tvd = []


def only_attention_cd_llama(attn, query_states, key_states, value_states, attention_mask, img_start, img_end):
    """cd attention output for the LAST query position in LLaMA architecture.
    Shapes: query_states (b, H, q, d), key_states/value_states (b, KV, k, d).
    """
    bsz = query_states.shape[0]
    q = query_states[:, :, -1:, :].float()
    num_kv_groups = getattr(attn, "num_key_value_groups", 1)
    k = repeat_kv(key_states, num_kv_groups).float()
    v = repeat_kv(value_states, num_kv_groups).float()
    kv_len = k.shape[-2]

    logits = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
    if attention_mask is not None:
        m = attention_mask[:, :, -1:, :kv_len]
        if m.dtype == torch.bool:
            logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
        else:
            logits = logits + m.float()
    w = nn.functional.softmax(logits, dim=-1)

    S, E = img_start, img_end
    w_cd = w.clone()
    zero = torch.tensor(0.0, device=w.device, dtype=w.dtype)

    # Text outlier removal (excluding BOS at index 0)
    text = torch.cat([w_cd[:, :, :, 1:S], w_cd[:, :, :, E:]], dim=-1)
    text = torch.where(text > text.mean(-1, keepdim=True) + text.std(-1, keepdim=True), zero, text)
    w_cd[:, :, :, 1:S] = text[:, :, :, :S - 1]
    w_cd[:, :, :, E:] = text[:, :, :, S - 1:]
    text_norm = (text / (text.sum(-1, keepdim=True) + 1e-12))[:, :, -1, :].unsqueeze(-2)

    # Image outlier removal
    img = w_cd[:, :, :, S:E]
    img = torch.where(img > img.mean(-1, keepdim=True) + img.std(-1, keepdim=True), zero, img)
    w_cd[:, :, :, S:E] = img
    img_norm = (img / (img.sum(-1, keepdim=True) + 1e-12))[:, :, -1, :].unsqueeze(-2)

    # Entropy calculation & head pruning
    entropy_text = -torch.sum(text_norm * torch.log(text_norm + 1e-6), dim=-1)
    entropy_img = -torch.sum(img_norm * torch.log(img_norm + 1e-6), dim=-1)
    entropy_text = torch.nan_to_num(entropy_text, nan=0.0)
    entropy_img = torch.nan_to_num(entropy_img, nan=float('inf'))
    ratio = (entropy_text.sum(-1) / (entropy_img.sum(-1) + 1e-12)).squeeze()
    
    if ratio.ndim == 0:
        ratio = ratio.unsqueeze(0)
    removed_heads = torch.where(ratio < ratio.mean())
    w_cd[:, removed_heads[0], :, :] = 0

    out = torch.matmul(w_cd, v).to(query_states.dtype)
    out = out.transpose(1, 2).reshape(bsz, 1, -1)
    return attn.o_proj(out)


def _make_patched_llama_attention_forward(attn, state):
    orig_forward = attn.forward

    def forward(*args, **kwargs):
        # Call original forward for the main output
        attn_output, attn_weights = orig_forward(*args, **kwargs)

        # Intercept and compute ONLY cd branch
        # hidden_states is either args[0] or in kwargs
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        position_embeddings = kwargs.get("position_embeddings")
        attention_mask = kwargs.get("attention_mask")
        past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value"))

        bsz, q_len, _ = hidden_states.size()
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        num_heads = attn.config.num_attention_heads if hasattr(attn, "config") else attn.num_heads
        num_kv_heads = attn.config.num_key_value_heads if hasattr(attn, "config") else getattr(attn, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        if position_embeddings is not None:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # past_key_values already contains the updated keys/values from orig_forward!
            if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > attn.layer_idx:
                key_states = past_key_values.key_cache[attn.layer_idx]
                value_states = past_key_values.value_cache[attn.layer_idx]
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) >= 2:
                key_states = past_key_values[0]
                value_states = past_key_values[1]

        state.cd = only_attention_cd_llama(
            attn, query_states, key_states, value_states, attention_mask,
            state.img_start, state.img_end
        )
        return attn_output, attn_weights

    return forward, orig_forward


class OnlyLlava:
    """Installs ONLY on a LlavaForConditionalGeneration (HuggingFace transformers >= 4.45.0).
    Use `set_image_span(img_start, img_end)` before `generate` and pass
    `logits_processor=LogitsProcessorList([OnlyLlavaLogitsProcessor(...)])` with do_sample=False.
    """

    def __init__(self, model, enhance_layer_index=0):
        self.model = model
        self.state = OnlyLlavaState()
        
        # Access language model components in LlavaForConditionalGeneration
        lm = model.language_model
        if hasattr(lm, "model"):
            layers = lm.model.layers
            self.norm = lm.model.norm
        else:
            layers = lm.layers
            self.norm = lm.norm
        self.lm_head = getattr(lm, "lm_head", None) or getattr(model, "lm_head", None)
        self.enhance_layer_index = enhance_layer_index
        self.last_index = len(layers) - 1

        attn = layers[enhance_layer_index].self_attn
        patched_fn, orig_fn = _make_patched_llama_attention_forward(attn, self.state)
        self._orig_forward = orig_fn
        self._attn = attn
        attn.forward = patched_fn

        self._handles = []
        last = layers[self.last_index]
        state = self.state

        def pre_hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            state.residual = hs[:, -1:, :]

        def post_hook(module, args, kwargs, output):
            if enhance_layer_index == self.last_index:
                state.cd_final = state.cd
                return
            # Multi-GPU safety: ensure state.cd is on same device as last layer
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

    def set_image_span(self, input_ids, image_token_id=None):
        """Set the image token span in the multimodal sequence.
        In transformers >= 4.45.0, LlavaForConditionalGeneration input_ids contains
        a contiguous run of image_token_id (e.g. 576 tokens for 24x24 patches).
        """
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_index", 32000)
        if isinstance(input_ids, torch.Tensor):
            pos = (input_ids[0] == image_token_id).nonzero().flatten()
            if pos.numel() > 0:
                start = pos[0].item()
                end = pos[-1].item() + 1
                self.state.img_start, self.state.img_end = start, end
                self.state.tvd = []
                return
        # Default fallback for standard LLaVA-v1 prompt template
        self.state.img_start = 35
        self.state.img_end = 35 + 576
        self.state.tvd = []

    def logits_cd(self):
        s = self.state
        device = self.lm_head.weight.device
        h_norm = s.h_norm
        cd_norm = self.norm(s.cd_final.to(self.norm.weight.device))
        s.h_norm = h_norm
        return self.lm_head(cd_norm.to(device) + 0.5 * h_norm.to(device))[:, -1, :]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._attn.forward = self._orig_forward


class OnlyLlavaLogitsProcessor(LogitsProcessor):
    """Contrastive decoding LogitsProcessor for LLaVA with ONLY intervention.
    Used with greedy decoding (do_sample=False).
    """

    def __init__(self, only: OnlyLlava, alpha_pos=3.0, alpha_neg=1.0, beta=0.1, gamma=0.2):
        self.only = only
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.beta = beta
        self.gamma = gamma

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
