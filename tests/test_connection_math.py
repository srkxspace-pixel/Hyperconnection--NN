"""The connection wiring matches the paper equations away from init (random, asymmetric weights).

Equivalence at init cannot catch a transposed mixing matrix because identity / 0.9I+0.1J/n are symmetric, so here the
forward is compared with explicit loops over [HC] Eq. (2) / [mHC] Eq. (3) built from the layer's own `weights()`.
"""
import pytest
import torch

from hcsmall.connections import make_connection

B, N, T, D, C = 2, 3, 4, 6, 5


def reference(conn, H, F, shortcut, layout):
    """H'_j = sum_i R[i,j] S(H_i) + w_write[j] F(sum_i w_read[i] H_i), with explicit loops. Weights are used as
    returned by conn.weights(): tokens [b,i,t] / [b,i,t,j]; conv [b,i] / [b,i,j]."""
    w_read, w_write, R = (w.expand(H.shape[0], *w.shape[1:]) for w in conn.weights(H))   # static weights: batch dim is 1
    n = H.shape[1]
    out = []
    for b in range(H.shape[0]):
        if layout == "tokens":
            h = sum(w_read[b, i, :, None] * H[b, i] for i in range(n))                    # (T, d)
            y = F(h[None])[0]
            Hs = [H[b, i] for i in range(n)]
            out.append(torch.stack([sum(R[b, i, :, j, None] * Hs[i] for i in range(n)) + w_write[b, j, :, None] * y
                                    for j in range(n)]))
        else:
            h = sum(w_read[b, i] * H[b, i] for i in range(n))                             # (C, h, w)
            y = F(h[None])[0]
            Hs = [shortcut(H[b, i][None])[0] if shortcut else H[b, i] for i in range(n)]
            out.append(torch.stack([sum(R[b, i, j] * Hs[i] for i in range(n)) + w_write[b, j] * y for j in range(n)]))
    return torch.stack(out)


def setup(kind, layout):
    torch.manual_seed(0)
    conv = layout == "conv"
    conn = make_connection(kind, N, C if conv else D, 2, layout)
    with torch.no_grad():
        for p in conn.parameters():
            p.add_(0.3 * torch.randn_like(p))                       # asymmetric, non-trivial weights
    dim = C if conv else D
    H = torch.randn(B, N, C, 4, 4) if conv else torch.randn(B, N, T, D)
    if conv:      # a stage transition: channels C -> 2C, stride 2, with a shared projection shortcut
        branch, shortcut = torch.nn.Conv2d(C, 2 * C, 3, 2, 1), torch.nn.Conv2d(C, 2 * C, 1, 2)
    else:
        branch, shortcut = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(D, D)), None
    return conn, H, branch, shortcut


@pytest.mark.parametrize("layout", ["tokens", "conv"])
@pytest.mark.parametrize("kind", ["shc", "dhc", "mhc"])
def test_forward_block_matches_explicit_loops(kind, layout):
    conn, H, branch, shortcut = setup(kind, layout)
    got = conn.forward_block(H, branch, shortcut)
    torch.testing.assert_close(got, reference(conn, H, branch, shortcut, layout), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("layout", ["tokens", "conv"])
def test_dhc_weights_of_stream_i_depend_only_on_stream_i(layout):
    """[HC] Eqs. (10)-(13): row i of [A_m | A_r] and B_i come from stream i's own (normalised) features."""
    conn, H, _, _ = setup("dhc", layout)
    base = conn.weights(H)
    H2 = H.clone()
    H2[:, 1] = torch.randn_like(H2[:, 1])                           # change stream 1 only
    new = conn.weights(H2)
    for a, b in zip(base[:2], new[:2]):                             # w_read, w_write: index [b, i, ...]
        assert not torch.allclose(a[:, 1], b[:, 1]) and torch.allclose(a[:, [0, 2]], b[:, [0, 2]])
    a, b = base[2], new[2]                                          # R: index [b, i(source), ..., j(target)]
    assert not torch.allclose(a[:, 1], b[:, 1]) and torch.allclose(a[:, [0, 2]], b[:, [0, 2]])
