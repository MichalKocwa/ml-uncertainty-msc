import numpy as np
import torch

from src.seeding import set_seed


def test_set_seed_reproducible_across_numpy_and_torch():
    set_seed(123)
    a_np = np.random.randn(5)
    a_torch = torch.randn(5)

    set_seed(123)
    b_np = np.random.randn(5)
    b_torch = torch.randn(5)

    np.testing.assert_array_equal(a_np, b_np)
    assert torch.equal(a_torch, b_torch)
