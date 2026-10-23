"""mHC mixing matrix is doubly stochastic; D16 init values; the unconstrained maps stay in range."""
import pytest
import torch

from hcsmall.connections import make_connection

N, DIM = 4, 8


def streams(layout):
    torch.manual_seed(0)
    return torch.randn(2, N, 5, DIM) if layout == "tokens" else torch.randn(2, N, DIM, 3, 3)


def as_matrix(R, layout):
    """R is [b, source, (t,) target] -> (..., source, target)."""
    return R.permute(0, 2, 1, 3) if layout == "tokens" else R


def maps(conn, layout):
    w_read, w_write, R = conn.weights(streams(layout))
    return w_read, w_write, as_matrix(R, layout)


@pytest.mark.parametrize("layout", ["tokens", "conv"])
def test_doubly_stochastic_away_from_init(layout):
    conn = make_connection("mhc", N, DIM, 2, layout)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in conn.parameters():
            p.add_(0.1 * torch.randn_like(p))
        conn.alpha.fill_(1.0)                      # make the input-dependent part matter
    w_read, w_write, R = maps(conn, layout)
    assert (R > 0).all()
    # R = (H^res)^T, so rows of H^res (sum over the source axis, -2) are normalised last: exact.
    torch.testing.assert_close(R.sum(-2), torch.ones_like(R.sum(-2)), atol=1e-12, rtol=0)
    # Columns of H^res (target axis, -1) are only approximately 1 at the paper's t_max = 20 ...
    torch.testing.assert_close(R.sum(-1), torch.ones_like(R.sum(-1)), atol=1e-2, rtol=0)
    # ... but the very same parameters converge to a truly doubly stochastic matrix with more iterations.
    conn.t_max = 2000
    _, _, R_conv = maps(conn, layout)
    torch.testing.assert_close(R_conv.sum(-1), torch.ones_like(R_conv.sum(-1)), atol=1e-9, rtol=0)
    torch.testing.assert_close(R_conv.sum(-2), torch.ones_like(R_conv.sum(-2)), atol=1e-12, rtol=0)
    assert ((w_read > 0) & (w_read < 1)).all() and ((w_write > 0) & (w_write < 2)).all()


@pytest.mark.parametrize("layout", ["tokens", "conv"])
@pytest.mark.parametrize("layer_idx", [0, 1, 5])
def test_d16_init_values(layout, layer_idx):
    conn = make_connection("mhc", N, DIM, layer_idx, layout)
    w_read, w_write, R = maps(conn, layout)
    torch.testing.assert_close(w_read.sum(1), torch.ones_like(w_read.sum(1)), atol=1e-12, rtol=0)   # sum over streams = 1
    assert (w_read[:, layer_idx % N] > 0.8).all()                                                  # rotating soft one-hot
    torch.testing.assert_close(w_write, torch.ones_like(w_write), atol=1e-12, rtol=0)              # H^post = 1
    expected = 0.9 * torch.eye(N) + 0.1 / N
    torch.testing.assert_close(R, expected.expand_as(R), atol=1e-12, rtol=0)                       # H^res = 0.9 I + 0.1 J / n
