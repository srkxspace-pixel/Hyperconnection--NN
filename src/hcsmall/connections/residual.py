"""The n = 1 baseline: x' = S(x) + F(x)."""
from typing import Callable, Optional

from torch import Tensor

from .base import Connection


class Residual(Connection):
    """Plain residual connection on (B, 1, ...) streams; layout independent."""

    def __init__(self) -> None:
        super().__init__(1)

    def forward_block(self, H: Tensor, F: Callable[[Tensor], Tensor],
                      shortcut: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        x = H[:, 0]
        return ((x if shortcut is None else shortcut(x)) + F(x)).unsqueeze(1)
