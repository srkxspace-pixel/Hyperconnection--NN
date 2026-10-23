"""Channel-first hyper-connections for conv streams (B, n, C, h, w): this project's adaptation."""
from torch import Tensor

from .hyper_connection import HyperConnection


class HyperConnectionConv(HyperConnection):
    """HC for H: (B, n, C, h, w). Every weight is a per-sample scalar broadcast over (C, h, w).

    D3(a): dynamic weights come from LayerNorm over channels of the spatially pooled streams,
    i.e. [HC] Eqs. (10)-(13) with the "token vector" replaced by GAP(H) in R^{B x n x C}.
    Parameter count per layer is the token formula with d = C.
    """

    READ, MIX, WRITE = "bi,bichw->bchw", "bij,bichw->bjchw", "bj,bchw->bjchw"

    def features(self, H: Tensor) -> Tensor:
        return self.norm(H.mean((3, 4)).unsqueeze(2))      # (B, n, 1, C): one "token" per sample

    def weights(self, H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        w_read, w_write, A_r = super().weights(H)           # (B|1, n, 1), (B|1, n, 1), (B|1, n, 1, n)
        return w_read[:, :, 0], w_write[:, :, 0], A_r[:, :, 0]
