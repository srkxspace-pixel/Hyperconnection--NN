"""Manifold-constrained hyper-connections (mHC) on token (B, n, T, d) and conv (B, n, C, h, w) streams.

[mHC] = Xie et al., arXiv:2512.24880. Equation numbers refer to docs/method_notes.md §8.
"""
import torch
from torch import Tensor, nn

from .base import Connection
from .hyper_connection import HyperConnection
from .hyper_connection_conv import HyperConnectionConv


def sinkhorn(logits: Tensor, t_max: int = 20) -> Tensor:
    """[mHC] Eq. (9): M0 = exp(logits), then t_max rounds of (column-normalise, row-normalise).

    The last two dims are (n, n). Rows sum to 1 exactly, columns to 1 up to the t_max truncation.
    """
    M = logits.exp()
    for _ in range(t_max):
        M = M / M.sum(-2, keepdim=True)
        M = M / M.sum(-1, keepdim=True)
    return M


class MHC(Connection):
    """mHC for H: (B, n, T, d). Per token, the n streams are flattened to one nd-vector.

    Eqs. (7)-(8): x' = RMSNorm(vec(H)); H^pre = sigmoid(alpha_0 x'phi_pre + b_pre);
    H^post = 2 sigmoid(alpha_1 x'phi_post + b_post); H^res = Sinkhorn(alpha_2 mat(x'phi_res) + b_res).
    phi is stored as one (n*d, n*n + 2n) matrix. The maps are computed in at least float32.

    Init (D16): phi = 0, so the dynamic part vanishes; b_post = 0 (H^post = 1); b_pre is the logit
    of a soft one-hot on stream (layer_idx mod n) with weights (1-(n-1)p_eps, p_eps, ...) summing to 1;
    b_res = log((1-lam) I + lam J / n), a Sinkhorn fixed point. At init this equals the residual model.
    """

    no_decay = ("alpha", "b_pre", "b_post", "b_res")      # D9/D17: phi (the dynamic matrix) is decayed
    READ, MIX, WRITE = HyperConnection.READ, HyperConnection.MIX, HyperConnection.WRITE

    def __init__(self, n: int, dim: int, layer_idx: int, t_max: int = 20, alpha0: float = 0.01,
                 eps_rms: float = 1e-20, p_eps: float = 0.05, lam: float = 0.1) -> None:
        super().__init__(n)
        self.dim, self.t_max, self.eps_rms = dim, t_max, eps_rms
        self.phi = nn.Parameter(torch.zeros(n * dim, n * n + 2 * n))
        self.alpha = nn.Parameter(torch.full((3,), alpha0))
        p = torch.full((n,), p_eps)
        p[layer_idx % n] = 1 - (n - 1) * p_eps
        self.b_pre = nn.Parameter(torch.log(p / (1 - p)))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.log((1 - lam) * torch.eye(n) + lam / n))

    def features(self, H: Tensor) -> Tensor:
        return H

    def weights(self, H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = self.n
        f = self.features(H)                                              # (B, n, T, d)
        dt = torch.promote_types(f.dtype, torch.float32)
        f = f.to(dt)
        xn = f * torch.rsqrt(f.pow(2).mean((1, 3), keepdim=True) + self.eps_rms)    # RMSNorm over (n, d)
        z = torch.einsum("bitd,idk->btk", xn, self.phi.to(dt).view(n, self.dim, -1))
        pre, post, res = z.split([n, n, n * n], dim=-1)
        a = self.alpha.to(dt)
        h_pre = torch.sigmoid(a[0] * pre + self.b_pre.to(dt))                       # (B, T, n)
        h_post = 2 * torch.sigmoid(a[1] * post + self.b_post.to(dt))                # (B, T, n)
        h_res = sinkhorn(a[2] * res.unflatten(-1, (n, n)) + self.b_res.to(dt), self.t_max)   # (B, T, target, source)
        return h_pre.transpose(1, 2), h_post.transpose(1, 2), h_res.permute(0, 3, 1, 2)    # R: [b, source, t, target]


class MHCConv(MHC):
    """mHC for conv streams (B, n, C, h, w); the maps come from vec(GAP(H)) in R^{B x nC} (D3a)."""

    READ, MIX, WRITE = HyperConnectionConv.READ, HyperConnectionConv.MIX, HyperConnectionConv.WRITE

    def features(self, H: Tensor) -> Tensor:
        return H.mean((3, 4)).unsqueeze(2)                                 # (B, n, 1, C)

    def weights(self, H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        w_read, w_write, R = super().weights(H)
        return w_read[:, :, 0], w_write[:, :, 0], R[:, :, 0]
