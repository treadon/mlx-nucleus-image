"""Nucleus-Image MoE DiT in MLX.

All dimensions verified against actual safetensors weight shapes.
"""

import math

import mlx.core as mx
import mlx.nn as nn


# ── RoPE ──

def build_rope_freqs(axes_dim, theta, max_len=4096):
    """Precompute RoPE cos/sin for positive AND negative positions.

    Matches NucleusMoEEmbedRope: pos_freqs for [0..4095], neg_freqs for [-4096..-1].
    """
    pos_cos, pos_sin = [], []
    neg_cos, neg_sin = [], []
    for dim in axes_dim:
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        # Positive positions: 0, 1, ..., max_len-1
        t_pos = mx.arange(max_len).astype(mx.float32)
        angles_pos = mx.outer(t_pos, freqs)
        pos_cos.append(mx.cos(angles_pos))
        pos_sin.append(mx.sin(angles_pos))
        # Negative positions: -max_len, ..., -2, -1
        t_neg = (mx.arange(max_len).astype(mx.float32)[::-1] * -1 - 1)
        angles_neg = mx.outer(t_neg, freqs)
        neg_cos.append(mx.cos(angles_neg))
        neg_sin.append(mx.sin(angles_neg))
    return pos_cos, pos_sin, neg_cos, neg_sin


def compute_image_rope(height, width, axes_dim, pos_cos, pos_sin, neg_cos, neg_sin, scale_rope=True):
    """Compute RoPE frequencies for image patches.

    With scale_rope=True (default for Nucleus), uses centered positions:
      height: [-ceil(H/2), ..., -1, 0, 1, ..., floor(H/2)-1]
      width:  [-ceil(W/2), ..., -1, 0, 1, ..., floor(W/2)-1]

    Returns (img_cos, img_sin) each of shape [H*W, D//2]
    """
    frame = 1
    # Frame axis: position 0
    f_cos = mx.broadcast_to(pos_cos[0][:frame], (frame, 1, 1, axes_dim[0] // 2))
    f_sin = mx.broadcast_to(pos_sin[0][:frame], (frame, 1, 1, axes_dim[0] // 2))
    f_cos = mx.broadcast_to(f_cos, (frame, height, width, axes_dim[0] // 2))
    f_sin = mx.broadcast_to(f_sin, (frame, height, width, axes_dim[0] // 2))

    if scale_rope:
        # Centered: negative positions + positive positions
        # Height: neg[-N_neg:] + pos[:N_pos] where N_neg = H - H//2, N_pos = H//2
        n_neg_h = height - height // 2
        n_pos_h = height // 2
        h_cos = mx.concatenate([neg_cos[1][-n_neg_h:], pos_cos[1][:n_pos_h]], axis=0)
        h_sin = mx.concatenate([neg_sin[1][-n_neg_h:], pos_sin[1][:n_pos_h]], axis=0)
        # Width
        n_neg_w = width - width // 2
        n_pos_w = width // 2
        w_cos = mx.concatenate([neg_cos[2][-n_neg_w:], pos_cos[2][:n_pos_w]], axis=0)
        w_sin = mx.concatenate([neg_sin[2][-n_neg_w:], pos_sin[2][:n_pos_w]], axis=0)
    else:
        h_cos = pos_cos[1][:height]
        h_sin = pos_sin[1][:height]
        w_cos = pos_cos[2][:width]
        w_sin = pos_sin[2][:width]

    h_cos = mx.broadcast_to(h_cos.reshape(1, height, 1, -1), (frame, height, width, axes_dim[1] // 2))
    h_sin = mx.broadcast_to(h_sin.reshape(1, height, 1, -1), (frame, height, width, axes_dim[1] // 2))
    w_cos = mx.broadcast_to(w_cos.reshape(1, 1, width, -1), (frame, height, width, axes_dim[2] // 2))
    w_sin = mx.broadcast_to(w_sin.reshape(1, 1, width, -1), (frame, height, width, axes_dim[2] // 2))

    img_cos = mx.concatenate([f_cos, h_cos, w_cos], axis=-1).reshape(frame * height * width, -1)
    img_sin = mx.concatenate([f_sin, h_sin, w_sin], axis=-1).reshape(frame * height * width, -1)
    return img_cos, img_sin


def compute_text_rope(max_txt_len, max_vid_index, axes_dim, pos_cos, pos_sin):
    """Compute RoPE frequencies for text tokens.

    Text positions start after max_vid_index (= max(H/2, W/2) with scale_rope).
    """
    start = max_vid_index
    cos_parts = []
    sin_parts = []
    for i, dim in enumerate(axes_dim):
        cos_parts.append(pos_cos[i][start:start + max_txt_len])
        sin_parts.append(pos_sin[i][start:start + max_txt_len])
    txt_cos = mx.concatenate(cos_parts, axis=-1)
    txt_sin = mx.concatenate(sin_parts, axis=-1)
    return txt_cos, txt_sin


def apply_rotary_emb_complex(x, freqs_cos, freqs_sin):
    """Apply rotary embeddings using complex multiplication.

    x: [B, S, H, D]
    freqs_cos, freqs_sin: [S, D//2] (real and imaginary parts of complex exponentials)

    Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    """
    # Reshape x into pairs: [B, S, H, D//2, 2]
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]  # [B, S, H, D//2]
    x_imag = x_pairs[..., 1]

    # Broadcast freqs: [1, S, 1, D//2]
    cos_ = freqs_cos[None, :, None, :].astype(x.dtype)
    sin_ = freqs_sin[None, :, None, :].astype(x.dtype)

    # Complex multiply
    out_real = x_real * cos_ - x_imag * sin_
    out_imag = x_real * sin_ + x_imag * cos_

    # Interleave back: [B, S, H, D//2, 2] → [B, S, H, D]
    out = mx.stack([out_real, out_imag], axis=-1)
    return out.reshape(x.shape)


def timestep_embedding(t, dim: int):
    half = dim // 2
    freqs = mx.exp(-math.log(10000) * mx.arange(0, half).astype(mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


# ── TimestepEmbedder: linear_1 [8192, 2048], linear_2 [2048, 8192] ──

class TimestepEmbedder(nn.Module):
    def __init__(self, in_dim: int, expand_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, expand_dim)
        self.linear_2 = nn.Linear(expand_dim, in_dim)

    def __call__(self, t):
        return self.linear_2(nn.silu(self.linear_1(t)))


# ── Gated projection (shared between dense FFN and shared expert) ──

class GatedProj(nn.Module):
    """net.0.proj: [hidden*2, in_dim]. SwiGLU: first_half * silu(second_half)."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim * 2, bias=False)

    def __call__(self, x):
        hidden, gate = mx.split(self.proj(x), 2, axis=-1)
        return hidden * nn.silu(gate)


# ── Dense FFN (layers 0-2) ──
# net.0.proj: [10752, 2048] → gated, hidden=5376
# net.2: [2048, 5376]

class DenseFFN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = [GatedProj(in_dim, hidden_dim), None, nn.Linear(hidden_dim, in_dim, bias=False)]

    def __call__(self, x):
        return self.net[2](self.net[0](x))


# ── MoE FFN (layers 3-31) ──

class SharedExpert(nn.Module):
    """shared_expert.net.0.proj: [2688, 2048], net.2: [2048, 1344]"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = [GatedProj(in_dim, hidden_dim), None, nn.Linear(hidden_dim, in_dim, bias=False)]

    def __call__(self, x):
        return self.net[2](self.net[0](x))


class Experts(nn.Module):
    """Packed expert weights.
    gate_up_proj: [64, in_dim, hidden*2] (weight shape from safetensors)
    down_proj: [64, hidden, in_dim]
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.gate_up_proj = mx.zeros((num_experts, in_dim, hidden_dim * 2))
        self.down_proj = mx.zeros((num_experts, hidden_dim, in_dim))


class MoEFFN(nn.Module):
    """gate: [64, 4096] — router input is concat(token, timestep)
    experts: packed, shared_expert: standard FFN

    Uses expert-choice routing: each expert picks its top-C tokens
    (capacity-based), matching the diffusers NucleusMoELayer.
    """

    def __init__(self, in_dim: int, expert_hidden_dim: int, num_experts: int = 64,
                 route_scale: float = 2.5, capacity_factor: float = 2.0):
        super().__init__()
        self.num_experts = num_experts
        self.route_scale = route_scale
        self.capacity_factor = capacity_factor

        # Router input is 2*in_dim (token + timestep concat)
        self.gate = nn.Linear(in_dim * 2, num_experts, bias=False)
        self.experts = Experts(in_dim, expert_hidden_dim, num_experts)
        self.shared_expert = SharedExpert(in_dim, expert_hidden_dim)

    def __call__(self, x, timestep_emb=None, unmodulated_x=None):
        """Expert-choice routing: each expert picks top-C tokens.

        C = ceil(capacity_factor * S / num_experts)
        Per-token gating normalization ensures tokens selected by
        multiple experts have their weights sum to route_scale.
        """
        B, S, D = x.shape

        # Decoupled routing: router sees unmodulated tokens + timestep
        if unmodulated_x is None:
            unmodulated_x = x
        if timestep_emb is not None:
            t_expanded = mx.broadcast_to(timestep_emb[:, None, :], (B, S, D))
            router_input = mx.concatenate([t_expanded, unmodulated_x], axis=-1)
        else:
            router_input = unmodulated_x

        logits = self.gate(router_input)  # [B, S, E]
        scores = mx.softmax(logits.astype(mx.float32), axis=-1).astype(x.dtype)

        # Expert-choice: transpose to [B, E, S], each expert scores all tokens
        affinity = mx.transpose(scores, (0, 2, 1))  # [B, E, S]
        capacity = max(1, math.ceil(self.capacity_factor * S / self.num_experts))

        # B=1 for inference — squeeze batch dim for simpler indexing
        aff = affinity[0]     # [E, S]
        x_flat = x[0]         # [S, D]

        # Each expert picks top-C tokens (argsort descending)
        sorted_idx = mx.argsort(-aff, axis=-1)[:, :capacity]  # [E, C]
        top_scores = mx.take_along_axis(aff, sorted_idx, axis=-1)  # [E, C]

        # Per-token normalization (tokens picked by multiple experts
        # have their scores normalized to sum to route_scale)
        flat_idx = sorted_idx.reshape(-1)       # [E*C]
        flat_scores = top_scores.reshape(-1)    # [E*C]

        # Indicator matrix for scatter operations: [E*C, S]
        indicator = (flat_idx[:, None] == mx.arange(S)[None, :]).astype(x.dtype)

        # Per-token score sums
        token_sums = indicator.T @ flat_scores   # [S]

        # Normalized gating weights
        flat_norm = flat_scores / (token_sums[flat_idx] + 1e-12) * self.route_scale

        # Gather selected tokens for all experts
        gathered = x_flat[flat_idx]  # [E*C, D]

        # Process through each expert's weights
        gu = self.experts.gate_up_proj  # [E, D, hidden*2]
        down = self.experts.down_proj   # [E, hidden, D]

        expert_outputs = []
        for e in range(self.num_experts):
            start = e * capacity
            end = start + capacity
            tokens_e = gathered[start:end]  # [C, D]

            h = tokens_e @ gu[e]  # [C, hidden*2]
            gh, uh = mx.split(h, 2, axis=-1)  # SwiGLUExperts: first=gate, second=up
            h = nn.silu(gh) * uh
            h = h @ down[e]  # [C, D]
            expert_outputs.append(h)

        all_expert_out = mx.concatenate(expert_outputs, axis=0)  # [E*C, D]
        all_expert_out = all_expert_out * flat_norm[:, None]       # weighted

        # Scatter back: indicator.T is [S, E*C], matmul gives [S, D]
        scattered = indicator.T @ all_expert_out

        # Shared expert (always runs on all tokens)
        shared_out = self.shared_expert(x)[0]  # [S, D]

        return (shared_out + scattered).reshape(1, S, D)


# ── Joint Attention ──

class JointAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.to_out = [nn.Linear(num_heads * head_dim, hidden_dim, bias=False)]

        self.add_k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.add_v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)

        self.norm_q = nn.RMSNorm(head_dim)
        self.norm_k = nn.RMSNorm(head_dim)
        self.norm_added_q = nn.RMSNorm(head_dim)
        self.norm_added_k = nn.RMSNorm(head_dim)

    def __call__(self, img_x, txt_kv, img_rope=None, txt_rope=None):
        B, S_img, _ = img_x.shape
        S_txt = txt_kv.shape[1]

        q = self.to_q(img_x).reshape(B, S_img, self.num_heads, self.head_dim)
        k = self.to_k(img_x).reshape(B, S_img, self.num_kv_heads, self.head_dim)
        v = self.to_v(img_x).reshape(B, S_img, self.num_kv_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        txt_k = self.add_k_proj(txt_kv).reshape(B, S_txt, self.num_kv_heads, self.head_dim)
        txt_v = self.add_v_proj(txt_kv).reshape(B, S_txt, self.num_kv_heads, self.head_dim)
        txt_k = self.norm_added_k(txt_k)

        # Apply RoPE
        if img_rope is not None:
            img_cos, img_sin = img_rope
            q = apply_rotary_emb_complex(q, img_cos, img_sin)
            k = apply_rotary_emb_complex(k, img_cos, img_sin)
        if txt_rope is not None:
            txt_cos, txt_sin = txt_rope
            txt_k = apply_rotary_emb_complex(txt_k, txt_cos, txt_sin)

        k = mx.concatenate([k, txt_k], axis=1)
        v = mx.concatenate([v, txt_v], axis=1)

        if self.num_kv_heads < self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, r, axis=2)
            v = mx.repeat(v, r, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim))
        out = out.transpose(0, 2, 1, 3).reshape(B, S_img, -1)
        return self.to_out[0](out)


# ── Transformer Block ──
# img_mod.1: [8192, 2048] → 4 modulations (shift_attn, scale_attn, shift_mlp, scale_mlp)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, dense_hidden: int, is_moe: bool,
                 num_experts: int, expert_hidden: int, route_scale: float,
                 text_dim: int, capacity_factor: float = 2.0):
        super().__init__()
        self.is_moe = is_moe

        self.attn = JointAttention(hidden_dim, num_heads, num_kv_heads, head_dim)

        if is_moe:
            self.is_moe = True
            self.img_mlp = MoEFFN(hidden_dim, expert_hidden, num_experts, route_scale, capacity_factor)
        else:
            self.is_moe = False
            self.img_mlp = DenseFFN(hidden_dim, dense_hidden)

        # Pre-norms: LayerNorm without affine
        self.pre_attn_norm = nn.LayerNorm(hidden_dim, affine=False)
        self.pre_mlp_norm = nn.LayerNorm(hidden_dim, affine=False)

        # 4 modulations: scale1, gate1, scale2, gate2
        self.img_mod = [nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 4)]
        self.encoder_proj = nn.Linear(text_dim, hidden_dim)

    def __call__(self, img_x, txt_kv, c, img_rope=None, txt_rope=None):
        # 4 modulations: scale1, gate1, scale2, gate2
        mod = self.img_mod[1](self.img_mod[0](c))[:, None, :]  # [B, 1, 4*H]
        scale1, gate1, scale2, gate2 = mx.split(mod, 4, axis=-1)

        # Clamp gates
        gate1 = mx.clip(gate1, -2.0, 2.0)
        gate2 = mx.clip(gate2, -2.0, 2.0)

        txt_projected = self.encoder_proj(txt_kv)

        # Attention: LayerNorm → scale → attn → tanh gate
        img_normed = self.pre_attn_norm(img_x)
        img_modulated = img_normed * (1 + scale1)
        attn_out = self.attn(img_modulated, txt_projected, img_rope=img_rope, txt_rope=txt_rope)
        img_x = img_x + mx.tanh(gate1) * attn_out

        # FFN: LayerNorm → scale → mlp → tanh gate
        img_normed2 = self.pre_mlp_norm(img_x)
        img_modulated2 = img_normed2 * (1 + scale2)
        if self.is_moe:
            mlp_out = self.img_mlp(img_modulated2, timestep_emb=c, unmodulated_x=img_normed2)
        else:
            mlp_out = self.img_mlp(img_modulated2)
        img_x = img_x + mx.tanh(gate2) * mlp_out

        return img_x


# ── Full Model ──

class NucleusMoEDiT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        num_heads = config.get("num_attention_heads", 16)
        head_dim = config.get("attention_head_dim", 128)
        hidden = num_heads * head_dim  # 2048
        num_kv_heads = config.get("num_key_value_heads", 4)
        num_layers = config.get("num_layers", 32)
        num_experts = config.get("num_experts", 64)
        expert_hidden = config.get("moe_intermediate_dim", 1344)
        route_scale = config.get("route_scale", 2.5)
        text_dim = config.get("joint_attention_dim", 4096)
        in_channels = config.get("in_channels", 64)
        out_channels = config.get("out_channels", 16)
        axes_dims = config.get("axes_dims_rope", [16, 56, 56])

        # Dense FFN hidden: from weight [10752, 2048] → 10752/2 = 5376
        dense_hidden = 5376

        self.hidden_dim = hidden
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.img_in = nn.Linear(in_channels, hidden)

        # Timestep: linear_1 [8192, 2048], linear_2 [2048, 8192]
        self.time_text_embed = {
            "timestep_embedder": TimestepEmbedder(hidden, 8192),
            "norm": nn.RMSNorm(hidden),
        }

        self.txt_norm = nn.RMSNorm(text_dim)
        self._axes_dim = axes_dims
        self._scale_rope = True  # Nucleus uses centered positions
        self._pos_cos, self._pos_sin, self._neg_cos, self._neg_sin = build_rope_freqs(axes_dims, 10000)

        capacity_factors = config.get("capacity_factors", [0.0] * 3 + [2.0] * 29)

        dense_layers = {0, 1, 2}
        self.transformer_blocks = [
            TransformerBlock(
                hidden, num_heads, num_kv_heads, head_dim, dense_hidden,
                is_moe=(i not in dense_layers),
                num_experts=num_experts, expert_hidden=expert_hidden,
                route_scale=route_scale, text_dim=text_dim,
                capacity_factor=capacity_factors[i],
            )
            for i in range(num_layers)
        ]

        # norm_out: AdaLayerNormContinuous(elementwise_affine=False, eps=1e-6)
        # Applies LayerNorm before adaptive modulation
        self.norm_out = {
            "norm": nn.LayerNorm(hidden, affine=False, eps=1e-6),
            "linear": nn.Linear(hidden, hidden * 2),
        }
        # proj_out: [64, 2048] → 64 = patch_size² * out_channels
        self.proj_out = nn.Linear(hidden, in_channels, bias=False)

    def __call__(self, hidden_states, timestep, txt_kv, grid_h=None, grid_w=None):
        B = hidden_states.shape[0]

        x = self.img_in(hidden_states)

        # Timesteps(scale=1000): multiply by 1000 then sinusoidal embed
        # Input timestep is already sigma*1000 from the scheduler
        t_emb = timestep_embedding(timestep * 1000, self.hidden_dim).astype(x.dtype)
        c = self.time_text_embed["timestep_embedder"](t_emb)
        c = self.time_text_embed["norm"](c)

        txt_kv = self.txt_norm(txt_kv)

        # Build RoPE: image patches are on a grid, text follows after
        N_img = hidden_states.shape[1]
        if grid_h is None or grid_w is None:
            # Fallback: assume square (works only for square images)
            grid_h = int(N_img ** 0.5)
            grid_w = N_img // grid_h
        assert grid_h * grid_w == N_img, f"Grid {grid_h}x{grid_w} != N_img {N_img}"
        img_cos, img_sin = compute_image_rope(
            grid_h, grid_w, self._axes_dim,
            self._pos_cos, self._pos_sin, self._neg_cos, self._neg_sin,
            scale_rope=self._scale_rope,
        )

        T_txt = txt_kv.shape[1]
        # With scale_rope, text starts at max(H/2, W/2)
        max_vid_idx = max(grid_h // 2, grid_w // 2) if self._scale_rope else max(grid_h, grid_w)
        txt_cos, txt_sin = compute_text_rope(T_txt, max_vid_idx, self._axes_dim, self._pos_cos, self._pos_sin)

        img_rope = (img_cos, img_sin)
        txt_rope = (txt_cos, txt_sin)

        for block in self.transformer_blocks:
            x = block(x, txt_kv, c, img_rope=img_rope, txt_rope=txt_rope)

        # AdaLayerNormContinuous: norm first, then adaptive modulation
        mod = self.norm_out["linear"](nn.silu(c))
        scale, shift = mx.split(mod, 2, axis=-1)  # scale first, shift second
        x = self.norm_out["norm"](x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.proj_out(x)

        return x
