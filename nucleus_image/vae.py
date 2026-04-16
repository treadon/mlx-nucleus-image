"""Qwen-Image VAE Decoder in MLX.

The original uses Conv3d (video-capable). For image generation T=1,
so all Conv3d reduce to Conv2d. We squeeze the temporal dimension
and use standard Conv2d.

Architecture (from config):
  z_dim: 16 (latent channels)
  base_dim: 96
  dim_mult: [1, 2, 4, 4] → channels [96, 192, 384, 384]
  num_res_blocks: 2
  8x spatial upscale

Weight naming (from safetensors):
  decoder.conv_in, decoder.mid_block.{resnets,attentions}, decoder.up_blocks,
  decoder.conv_out, decoder.norm_out
  Uses 'gamma' for norm weights (not 'weight')
  Conv weights are 5D: [out, in, D, H, W] → squeeze D for Conv2d
"""

import mlx.core as mx
import mlx.nn as nn


class RMSNorm2D(nn.Module):
    """RMS normalization with spatial gamma. Matches 'gamma' weight naming."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = mx.ones((channels,))

    def __call__(self, x):
        # x: [B, H, W, C]
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * self.gamma


class ResnetBlock(nn.Module):
    """Residual block. Matches: norm1.gamma, conv1, norm2.gamma, conv2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = RMSNorm2D(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = RMSNorm2D(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.conv_shortcut = None

    def __call__(self, x):
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """Self-attention. Matches: norm.gamma, to_qkv.{weight,bias}, proj.{weight,bias}.
    Note: weights are stored as Conv2d 1x1 [out, in, 1, 1] but we use Linear."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2D(channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.to_qkv(x)  # [B, H, W, C*3]
        qkv = qkv.reshape(B, H * W, C * 3)
        q, k, v = mx.split(qkv, 3, axis=-1)

        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        out = out.reshape(B, H, W, C)
        out = self.proj(out)
        return out + residual


class PixelShufflePlaceholder(nn.Module):
    """Placeholder for resample[0] (pixel unshuffle). Not used at inference."""
    pass


class Upsample(nn.Module):
    """2x spatial upsample. Matches: upsamplers.0.resample.{0,1} + [time_conv]."""

    def __init__(self, in_ch: int, out_ch: int, has_time_conv: bool = True):
        super().__init__()
        self.resample = [PixelShufflePlaceholder(), nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if has_time_conv:
            self.time_conv = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1)

    def __call__(self, x):
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        return self.resample[1](x)


class UpBlock(nn.Module):
    """Matches: resnets.{0,1,2}, [upsamplers.0]."""

    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int = 3, upsample_out_ch: int = None, has_time_conv: bool = True):
        super().__init__()
        self.resnets = [ResnetBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(num_res_blocks)]
        if upsample_out_ch is not None:
            self.upsamplers = [Upsample(out_ch, upsample_out_ch, has_time_conv=has_time_conv)]
        else:
            self.upsamplers = None

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class MidBlock(nn.Module):
    """Matches: resnets.{0,1}, attentions.0."""

    def __init__(self, channels: int):
        super().__init__()
        self.resnets = [ResnetBlock(channels, channels), ResnetBlock(channels, channels)]
        self.attentions = [AttentionBlock(channels)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Decoder(nn.Module):
    """Full decoder. Matches diffusers naming."""

    def __init__(self):
        super().__init__()
        # Config: base_dim=96, dim_mult=[1,2,4,4] → [96, 192, 384, 384]
        # Decoder goes reversed: [384, 384, 192, 96]
        self.conv_in = nn.Conv2d(16, 384, kernel_size=3, padding=1)
        self.mid_block = MidBlock(384)
        # From weight shapes:
        # block 0: all 384→384, upsample 384→192
        # block 1: first resnet 192→384 (has conv_shortcut), rest 384→384, upsample 384→192
        # block 2: all 192→192, upsample 192→96
        # block 3: first resnet 96→96, rest 96→96, no upsample
        self.up_blocks = [
            UpBlock(384, 384, upsample_out_ch=192),   # 384→384, upsample→192
            UpBlock(192, 384, upsample_out_ch=192),   # 192→384 (shortcut), upsample→192
            UpBlock(192, 192, upsample_out_ch=96, has_time_conv=False),  # 192→192, upsample→96 (no time_conv)
            UpBlock(96, 96, upsample_out_ch=None),    # 96→96, no upsample
        ]
        self.norm_out = RMSNorm2D(96)
        self.conv_out = nn.Conv2d(96, 3, kernel_size=3, padding=1)

    def __call__(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = nn.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x


class VAEDecoder(nn.Module):
    """Top-level: post_quant_conv + decoder."""

    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(16, 16, kernel_size=1)
        self.decoder = Decoder()

    def __call__(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
