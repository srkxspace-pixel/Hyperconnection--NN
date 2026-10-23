import pytest
import torch


@pytest.fixture(autouse=True)
def float64():
    """Equivalence tests run in float64; restore the default afterwards."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)
