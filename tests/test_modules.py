import pytest
import torch as t
import torch.nn as nn

from model.modules import ReLU, Linear

atol = 1e-5

@pytest.mark.parametrize(
        "input, expected",
        [
            (t.tensor([-1.0, -3.0, 5.3, 7.5]),t.tensor([0.0, 0.0, 5.3, 7.5])),
            (t.tensor([0.0, 0.0]),t.tensor([0.0, 0.0])),
            (t.tensor([-1.0, -3.0]),t.tensor([0.0, 0.0])),
            (t.tensor([1.0, 3.0]),t.tensor([1.0, 3.0]))
        ]
)
def test_relu(input, expected):
    relu = ReLU()
    out = relu(input)

    assert t.allclose(expected, out, atol=atol)

    print("test_relu() passed!")



def test_linear_no_bias_forward():
    x = t.rand((10, 512))

    lin = Linear(512, 64, bias=False)
    official = nn.Linear(512, 64, bias=False)

    # ensures weights are the same
    lin.load_state_dict(official.state_dict())

    actual = lin(x)
    expected = official(x)

    assert t.allclose(actual, expected, atol=atol)
    print("test_linear_no_bias_forward() passed!")   


def test_linear_with_bias_forward():
    x = t.rand((10, 512))

    lin = Linear(512, 64, bias=True)
    official = nn.Linear(512, 64, bias=True)

    # ensures weights are the same
    lin.load_state_dict(official.state_dict())

    actual = lin(x)
    expected = official(x)

    assert t.allclose(actual, expected, atol=atol)
    print("test_linear_with_bias_forward() passed!")