"""Exact upstream comparisons must reject representation changes."""

import numpy as np
import pytest
import torch

from tools.verify_sheetsage2 import in_memory_differences


@pytest.mark.parametrize(
    "right",
    [
        np.array([1, 2], dtype=np.float32),
        np.array([[1, 2]], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        [1, 2],
    ],
    ids=["dtype", "shape", "value", "type"],
)
def test_array_representation_mismatch_is_reported(right):
    differences = in_memory_differences(
        {"output": np.array([1, 2], dtype=np.int32)}, {"output": right}
    )
    assert len(differences) == 1
    assert differences[0]["path"] == "$.output"


def test_equal_array_is_accepted():
    left = np.array([1, 2], dtype=np.int32)
    assert in_memory_differences(left, left.copy()) == []


@pytest.mark.parametrize(
    "right",
    [
        torch.tensor([1, 2], dtype=torch.float32),
        torch.tensor([[1, 2]], dtype=torch.int32),
        torch.tensor([1, 3], dtype=torch.int32),
    ],
    ids=["dtype", "shape", "value"],
)
def test_tensor_comparison_remains_exact(right):
    left = torch.tensor([1, 2], dtype=torch.int32)
    assert in_memory_differences(left, right)
    assert in_memory_differences(left, left.clone()) == []
