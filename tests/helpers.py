"""Tiny layer stacks shared by the connection tests. Every arm reuses the same branch modules."""
import torch
from torch import nn

from hcsmall.connections import Connection, make_connection

B, T, D, C, L = 3, 6, 16, 8, 6


class TokenLayer(nn.Module):
    """Pre-Norm MLP sublayer: the norm lives inside the branch."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, x):
        return self.net(x)


class ConvLayer(nn.Module):
    """Pre-act basic-block branch (BN-ReLU-conv-BN-ReLU-conv); may change channels / stride."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(), nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False))

    def forward(self, x):
        return self.net(x)


def build(layout: str):
    """(branches, shortcuts, dims, x). The conv stack has a stage transition (C -> 2C, stride 2) at layer 3."""
    torch.manual_seed(0)
    if layout == "tokens":
        return [TokenLayer() for _ in range(L)], [None] * L, [D] * L, torch.randn(B, T, D)
    branches, shortcuts, dims = [], [], []
    for k in range(L):
        cin = C if k <= 3 else 2 * C
        cout = 2 * C if k >= 3 else C
        stride = 2 if k == 3 else 1
        branches.append(ConvLayer(cin, cout, stride))
        shortcuts.append(nn.Conv2d(cin, cout, 1, stride, bias=False) if k == 3 else None)
        dims.append(cin)
    return branches, shortcuts, dims, torch.randn(B, C, 8, 8)


def run(kind: str, n: int, layout: str, parts, perturb: float = 0.0):
    """Forward the stack; returns (reduced output, connections). `perturb` leaves the init point."""
    branches, shortcuts, dims, x = parts
    conns = [make_connection(kind, n, dims[k], k, layout) for k in range(L)]
    if perturb:
        with torch.no_grad():
            for c in conns:
                for p in c.parameters():
                    p.add_(perturb * torch.randn_like(p))
    H = Connection.expand(x, n)
    for conn, f, s in zip(conns, branches, shortcuts):
        H = conn.forward_block(H, f, s)
    return Connection.reduce(H), conns
