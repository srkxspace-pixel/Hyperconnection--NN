"""Static (SHC) and dynamic (DHC) hyper-connections on token streams (B, n, T, d).

[HC] = Zhu et al., arXiv:2409.19606. Equation numbers refer to docs/method_notes.md §2-4.
"""
import torch
from torch import Tensor, nn

from .base import Connection


class HyperConnection(Connection):
    """HC for H: (B, n, T, d). Streams on dim 1; weights are per token when dynamic.

    Static parameters (Eq. 1): a_m (n,) = A_m, a_r (n, n) = A_r, b (n,) = B, initialised by
    Eq. (14): A_m = e_{layer_idx mod n}, A_r = I, B = 1.
    Dynamic extras (Eqs. 10-13), all zero-initialised except the two 0.01 scales:
    w_m, w_b (d,), w_r (d, n) and scalars s_alpha, s_beta. The norm is a non-affine LayerNorm
    (the parameter count in [HC] App. B has no norm parameters).
    """

    no_decay = ("a_m", "a_r", "b", "s_alpha", "s_beta")   # D9/D17: static parameters are not decayed
    READ, MIX, WRITE = "bit,bitd->btd", "bitj,bitd->bjtd", "bjt,btd->bjtd"

    def __init__(self, n: int, dim: int, layer_idx: int, dynamic: bool) -> None:
        super().__init__(n)
        a_m = torch.zeros(n)
        a_m[layer_idx % n] = 1.0
        self.a_m, self.a_r, self.b = nn.Parameter(a_m), nn.Parameter(torch.eye(n)), nn.Parameter(torch.ones(n))
        self.dynamic = dynamic
        if dynamic:
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.w_m, self.w_b = nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))
            self.w_r = nn.Parameter(torch.zeros(dim, n))
            self.s_alpha, self.s_beta = nn.Parameter(torch.tensor(0.01)), nn.Parameter(torch.tensor(0.01))

    def features(self, H: Tensor) -> Tensor:
        """The per-stream vectors the dynamic weights are computed from, (B, n, T', d): Eq. (10)."""
        return self.norm(H)

    def weights(self, H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # shapes [b, i, t, j]: i = source stream, j = target stream; static ones broadcast over b and t
        a_m, b, a_r = self.a_m[None, :, None], self.b[None, :, None], self.a_r[None, :, None, :]
        if not self.dynamic:
            return a_m, b, a_r
        f = self.features(H)
        return (a_m + self.s_alpha * torch.tanh(f @ self.w_m),    # Eq. (12)
                b + self.s_beta * torch.tanh(f @ self.w_b),       # Eq. (11)
                a_r + self.s_alpha * torch.tanh(f @ self.w_r))    # Eq. (13)
