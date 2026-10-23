"""Layer connections: residual, static/dynamic HC, mHC; token and channel-first layouts."""
from .base import Connection
from .hyper_connection import HyperConnection
from .hyper_connection_conv import HyperConnectionConv
from .mhc import MHC, MHCConv
from .residual import Residual


def make_connection(kind: str, n: int, dim: int, layer_idx: int, layout: str) -> Connection:
    """kind: residual | shc | dhc | mhc. layout: 'tokens' (B,n,T,d) or 'conv' (B,n,C,h,w).

    `dim` is the feature size of the streams entering the layer (d, or C of the block input);
    `layer_idx` is the index of the layer in the network (sets the rotating read stream).
    """
    conv = layout == "conv"
    if kind == "residual":
        return Residual()
    if kind in ("shc", "dhc"):
        return (HyperConnectionConv if conv else HyperConnection)(n, dim, layer_idx, dynamic=kind == "dhc")
    if kind == "mhc":
        return (MHCConv if conv else MHC)(n, dim, layer_idx)
    raise ValueError(f"unknown connection kind: {kind!r}")
