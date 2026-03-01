# spectrum to vector and vector to spectrum modules


from ..utils.config import Config
import torch.nn as nn


class SpectrumEmbedder(nn.Module):
    def __init__(self, config: Config
