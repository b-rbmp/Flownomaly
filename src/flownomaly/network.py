"""Time-conditioned velocity network used by Flownomaly."""

from __future__ import annotations

import math
import torch
from torch import nn

from .protocol import PROTOCOL


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = math.log(10000) / (half - 1)
        frequencies = torch.exp(torch.arange(half, device=time.device) * -frequencies)
        embedding = time * frequencies.unsqueeze(0)
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, dimension: int, time_dimension: int):
        super().__init__()
        hidden = dimension * PROTOCOL.mlp_ratio
        self.normalization = nn.LayerNorm(dimension, elementwise_affine=False)
        self.modulation = nn.Linear(time_dimension, 2 * dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self, state: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        residual = self.normalization(state)
        residual = residual * (1.0 + scale) + shift
        return state + self.mlp(residual)


class VelocityNetwork(nn.Module):
    def __init__(self, embedding_dimension: int = 1536):
        super().__init__()
        time_dimension = PROTOCOL.time_embedding_dim
        bottleneck = PROTOCOL.bottleneck_dim
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dimension),
            nn.Linear(time_dimension, time_dimension),
            nn.GELU(),
        )
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dimension + time_dimension, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            TimeConditionedResidualBlock(bottleneck, time_dimension)
            for _ in range(PROTOCOL.residual_blocks)
        )
        self.output_normalization = nn.LayerNorm(bottleneck)
        self.output_projection = nn.Linear(bottleneck, embedding_dimension)
        self.input_gate = nn.Sequential(
            nn.Linear(time_dimension, time_dimension),
            nn.GELU(),
            nn.Linear(time_dimension, 1),
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.input_gate[-1].weight)
        nn.init.zeros_(self.input_gate[-1].bias)

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(time)
        hidden = self.input_projection(torch.cat((state, time_embedding), dim=-1))
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        velocity = self.output_projection(self.output_normalization(hidden))
        return velocity + self.input_gate(time_embedding) * state
