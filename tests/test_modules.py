import pytest
import torch as t
import torch.nn as nn

from models.modules import ReLU, Linear, MLP

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

def test_mlp():
    """
    Testing using a simple MLP from PyTorch library
    """

    mlp = MLP()

    # random test with shape [b 28 28]
    x = t.rand((8, 28, 28))

    # create pytorch model
    class PyTorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear1 = nn.Linear(28*28, 100)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(100, 10)
        
        def forward(self, x):
            out = self.flatten(x)
            return self.linear2(self.relu(self.linear1(out)))

    official =  PyTorMLP()

    mlp.load_state_dict(official.state_dict())

    actual = mlp(x)
    expected = official(x)

    assert t.allclose(actual, expected, atol=atol)

    print("test_mlp() passed!")