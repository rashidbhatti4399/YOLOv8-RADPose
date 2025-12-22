import torch
import torch.nn as nn
import torch.nn.functional as F

class RADPoseBlock(nn.Module):
    def __init__(self, c=256, num_heads=8, pos_embed=True):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        self.pos_embed_enabled = pos_embed

        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * 6),
            nn.GELU(),
            nn.Linear(c * 6, c)
        )
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        # Default positional embedding initialized for 8x8 (can be interpolated)
        self.register_parameter("pos_embed", nn.Parameter(torch.zeros(1, 64, c)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        N = H * W

        # Handle positional embedding
        if self.pos_embed_enabled:
            pos_embed = self.pos_embed

            if N != pos_embed.shape[1]:
                side_old = int(self.pos_embed.shape[1] ** 0.5)
                pos_embed_2d = pos_embed.reshape(1, side_old, side_old, self.c).permute(0, 3, 1, 2)
                pos_embed_2d = F.interpolate(pos_embed_2d, size=(H, W), mode='bilinear', align_corners=False)
                pos_embed = pos_embed_2d.flatten(2).permute(0, 2, 1)  # (1, H*W, C)

            x_flat = x_flat + pos_embed

        # Multi-head self-attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_attn = self.norm1(x_flat + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x_attn)
        x_ffn = self.norm2(x_attn + ffn_out)

        # Reshape to (B, C, H, W)
        x_out = x_ffn.permute(0, 2, 1).reshape(B, C, H, W)

        # Residual connection
        return x + x_out

    def forward_fuse(self, x):
        return self.forward(x)
