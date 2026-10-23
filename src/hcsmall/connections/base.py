"""Common interface of every layer connection: residual (n = 1), HC, mHC.

Streams live on dim 1, so H has shape (B, n, ...). One Connection serves one layer and
owns that layer's weights; the layer function F and the shortcut S are passed in, so only
the connection differs between arms (one variable at a time).
"""
from typing import Callable, Optional

import torch
from torch import Tensor, nn


class Connection(nn.Module):
    """Per-layer stream connection.

    Subclasses provide `weights(H) -> (w_read, w_write, R)` and the three einsum strings of
    their tensor layout. With S the shortcut (identity if None):

        h_in = sum_i w_read_i * H_i
        H'_j = sum_i R_ij * S(H_i) + w_write_j * F(h_in)

    This is [HC] Eq. (2) with w_read = A_m, w_write = B, R = A_r, and [mHC] Eq. (3) with
    w_read = H^pre, w_write = H^post, R = (H^res)^T.
    """

    no_decay: tuple = ()   # parameter names exempt from weight decay (D9/D17)
    READ = MIX = WRITE = ""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    @staticmethod
    def expand(x: Tensor, n: int) -> Tensor:
        """(B, ...) -> (B, n, ...): n identical copies (a view)."""
        return x.unsqueeze(1).expand(-1, n, *x.shape[1:])

    @staticmethod
    def reduce(H: Tensor) -> Tensor:
        """(B, n, ...) -> (B, ...): mean over streams (D14). The paper's sum is n times this."""
        return H.mean(1)

    def weights(self, H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def forward_block(self, H: Tensor, F: Callable[[Tensor], Tensor],
                      shortcut: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """One layer. `shortcut` maps one stream (B, C, h, w) to the new shape at a ResNet
        stage transition; it is shared by all streams (D4a) and sees the raw input (D15)."""
        w_read, w_write, R = self.weights(H)
        y = F(torch.einsum(self.READ, w_read, H))
        Hs = H if shortcut is None else shortcut(H.flatten(0, 1)).unflatten(0, H.shape[:2])
        return torch.einsum(self.MIX, R, Hs) + torch.einsum(self.WRITE, w_write, y)
