import torch as t
import einops
import math
import torch.nn as nn

from torch import Tensor

class ReLU(nn.Module):
    """
    ReLU activation function

    Has no parameters
    """
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(t.tensor(0.0), x)

class Linear(nn.Module):
    """
    Linear layer implementation with Uniform Kaiming initialization
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = t.rand(out_features, in_features)

        # kaiming init
        scaling_factor = math.sqrt(in_features)
        weight = scaling_factor*(2*weight - 1)
        self.weight = nn.Parameter(weight)

        # bias init
        if bias:
            bias = scaling_factor*(2*t.rand(out_features) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward layer of linear:

        out = x*self.weight^{T} + self.bias

        
        Args:
            x (Tensor): data with shape [batch input_features]

        Returns:
            Tensor: output of linear model with shape [batch output_features]
        """

        out = einops.einsum(
            x,
            self.weight,
            "... i, o i -> ... o"
        )

        if self.bias is not None:
            out += self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

# `Flatten` from Callum McDougall's ARENA curriculum
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right) # type: ignore

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


class MLP(nn.Module):
    # TODO: update so can take arbitrary inputs, outputs, and number of hidden layers
    # TODO: at that point will probably need a separate mlp.py file
    """
    Simple MLP w/ two hidden layers, for MNIST

    Model architecture
    Input: bx28x28 -> shape [batch 28 28]
    Linear: 28**2 inputs, 100 outputs -> shape [batch 28**2 100]
    ReLU
    Linear: 100 inputs, 10 outputs -> shape [batch 100 10]
    """
    def __init__(self):
        super().__init__()

        # naming conventions to match PyTorch
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=28**2, out_features=100)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=100, out_features=10)


    def forward(self, x: Tensor) -> Tensor:
        """
        Returns logits of size 10

        Args:
            x (Tensor): Shape [batch 28 28]
        
        Returns:
            logits (Tensor): Shape [batch 10]
        """

        # keep batch dimension
        x_f = self.flatten(x)
        hid = self.relu(self.linear1(x_f))

        logits = self.linear2(hid)

        return logits

        
