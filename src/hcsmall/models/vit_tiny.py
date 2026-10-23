"""Pre-Norm ViT-Tiny with a pluggable stream connection (D5). Each attention and each MLP sublayer is one layer."""
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..connections import Connection, make_connection


class Attention(nn.Module):
    """LayerNorm then multi-head self-attention (the norm is part of the branch, as in Pre-Norm)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.norm = heads, nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D))


class Mlp(nn.Module):
    """LayerNorm then Linear-GELU-Linear."""

    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        self.norm, self.fc1, self.fc2 = nn.LayerNorm(dim), nn.Linear(dim, ratio * dim), nn.Linear(ratio * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class ViTTiny(nn.Module):
    """Patch embedding + class token + learned positions, `depth` blocks (2*depth connection layers),
    final LayerNorm, linear head on the class token. D5: dim 192, 3 heads, MLP ratio 4, patch 4 on 32x32.
    With img_size=224, patch=16, num_classes=1000 it has the standard ViT-Tiny parameter count."""

    def __init__(self, depth: int, num_classes: int, connection: str = "residual", n: int = 1,
                 img_size: int = 32, patch: int = 4, dim: int = 192, heads: int = 3, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.n = n
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        tokens = (img_size // patch) ** 2 + 1
        self.cls_token, self.pos_embed = nn.Parameter(torch.zeros(1, 1, dim)), nn.Parameter(torch.zeros(1, tokens, dim))
        self.branches = nn.ModuleList([m for _ in range(depth) for m in (Attention(dim, heads), Mlp(dim, mlp_ratio))])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.conns = nn.ModuleList([make_connection(connection, n, dim, k, "tokens") for k in range(2 * depth)])

    def forward(self, x: Tensor, trace: Optional[Callable[[Tensor], None]] = None) -> Tensor:
        """`trace`, if given, is called with the reduced hidden state before layer 0 and after every layer."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        H = Connection.expand(x, self.n)
        if trace is not None:
            trace(Connection.reduce(H))
        for conn, f in zip(self.conns, self.branches):
            H = conn.forward_block(H, f)
            if trace is not None:
                trace(Connection.reduce(H))
        return self.head(self.norm(Connection.reduce(H))[:, 0])
