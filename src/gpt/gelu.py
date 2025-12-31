from typing import Self

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    GELU = "Gaussian error linear unit" offers improved performance for deep
    learning models, unlike the simpler ReLU. The exact version is defined as
    GELU(x) = x * phi(x), where phi = cdf of the standard Gaussian
    distribution. Here, we use a computationally cheaper approximation.
    """

    def __init__(self: Self) -> None:
        super().__init__()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
