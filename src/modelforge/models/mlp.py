from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, inputs: int, outputs: int, activation: str, dropout: float, residual: bool):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, outputs), getattr(nn, activation)(),
                                    nn.Dropout(dropout))
        self.skip = (nn.Identity() if inputs == outputs else nn.Linear(inputs, outputs, bias=False)) if residual else None

    def forward(self, x):
        result = self.layers(x)
        return result + self.skip(x) if self.skip is not None else result


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], activation: str,
                 dropout: float, residual: bool = False):
        super().__init__()
        layers = []
        size = input_size
        for width in hidden_sizes:
            layers.append(Block(size, width, activation, dropout, residual))
            size = width
        layers.append(nn.Linear(size, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = MLP(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
