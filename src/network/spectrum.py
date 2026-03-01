# spectrum to vector and vector to spectrum modules


from math import ceil
from ..utils.config import Config
import torch.nn as nn
import torch


class SpecEmbed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        num_features = config.spec.num_features // 4
        m_min = config.spec.m_min
        m_max = config.spec.m_max

        lows = num_features // 2
        highs = num_features - lows - 1

        b = torch.tensor(
            [(1 / (i * ceil(m_max / lows))) for i in range(lows, 0, -1)] + [1] +
            [(i * ceil(1 / m_min / highs)) for i in range(1, highs + 1, 1)]
        )
        self.b = b.unsqueeze(0)

    def forward(self, mz: torch.Tensor, i: torch.Tensor):
        mz = 2 * torch.pi * mz.unsqueeze(0).T @ self.b
        i = 2 * torch.pi * i.unsqueeze(0).T @ self.b
        return torch.cat([torch.sin(mz), torch.cos(mz), torch.sin(i), torch.cos(i)], dim=-1)
