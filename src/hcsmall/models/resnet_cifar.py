"""Pre-activation ResNet-(6k+2) for 32x32 inputs, with a pluggable stream connection (D1, D4a, D15a)."""
from typing import Callable, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from ..connections import Connection, make_connection


class Branch(nn.Module):
    """BN-ReLU-conv3x3(stride)-BN-ReLU-conv3x3: the residual function of a pre-act basic block."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(F.relu(self.bn2(self.conv1(F.relu(self.bn1(x))))))


class PreActResNet(nn.Module):
    """He et al. (ECCV 2016) pre-act ResNet for CIFAR: stem conv, 3 stages of k blocks (16/32/64 channels),
    final BN-ReLU, global average pool, linear head. Depth = 6k + 2.

    At the two stage transitions the shortcut is a 1x1 stride-2 conv on the *raw* block input (D15a),
    shared by all streams (D4a). The connection kind and stream count n change nothing else: the
    backbone consumes the same random numbers whatever the connection, so equal seeds give equal backbones.
    """

    def __init__(self, depth: int, num_classes: int, connection: str = "residual", n: int = 1,
                 widths: tuple = (16, 32, 64)) -> None:
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6k+2"
        k = (depth - 2) // 6
        self.n = n
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        branches, shortcuts, dims, cin = [], [], [], widths[0]
        for stage, c in enumerate(widths):
            for i in range(k):
                stride = 2 if (stage > 0 and i == 0) else 1
                branches.append(Branch(cin, c, stride))
                shortcuts.append(nn.Conv2d(cin, c, 1, stride, bias=False) if (stride != 1 or cin != c) else None)
                dims.append(cin)
                cin = c
        self.branches, self.shortcuts = nn.ModuleList(branches), nn.ModuleList(shortcuts)
        self.bn, self.fc = nn.BatchNorm2d(widths[-1]), nn.Linear(widths[-1], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # connections own no random numbers, so they are built last and never change the backbone init
        self.conns = nn.ModuleList([make_connection(connection, n, d, j, "conv") for j, d in enumerate(dims)])

    def forward(self, x: Tensor, trace: Optional[Callable[[Tensor], None]] = None) -> Tensor:
        """`trace`, if given, is called with the reduced hidden state before layer 0 and after every layer."""
        H = Connection.expand(self.stem(x), self.n)
        if trace is not None:
            trace(Connection.reduce(H))
        for conn, f, s in zip(self.conns, self.branches, self.shortcuts):
            H = conn.forward_block(H, f, s)
            if trace is not None:
                trace(Connection.reduce(H))
        return self.fc(F.relu(self.bn(Connection.reduce(H))).mean((2, 3)))
