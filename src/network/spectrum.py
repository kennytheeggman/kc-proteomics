# spectrum to vector and vector to spectrum modules
# !!! ignore all premz code i misunderstood data oops, it's also incomplete

from math import ceil
from ..utils.config import Config
import torch.nn as nn
import torch
from ..network.linear import FeedForward

class SpecEmbed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # initialize variables for fourier embedding
        num_peaks = config.spec.num_peaks
        m_min = config.spec.m_min
        m_max = config.spec.m_max
        k = int(1/m_min)
        len_b = int(m_max + k)
        self.num_peaks = num_peaks
        self.len_b = len_b

        # initialize network variables
        dp = config.spec.dp
        dm = config.spec.dm
        dropout_fourier = config.spec.dropout_fourier
        dropout_raw = config.spec.dropout_raw
        dropout_premz = config.spec.dropout_premz
        hidden_fourier = config.spec.hidden_fourier
        hidden_raw = config.spec.hidden_raw
        hidden_premz = config.spec.hidden_premz

        self.dp = dp
        self.dm = dm
        self.sz = dm + dp

        # intialize arrays, apparently they have to be these register buffer things?
        self.register_buffer('premz_encode', torch.empty((2*len_b + 1, 1)))  # first element will be the raw encoding
        b = torch.cat((torch.arange(m_max, 0, -1), m_min*torch.arange(k, 0, -1)))
        self.register_buffer('b', b)

        # set up feed forward networks
        self.ff_fourier = FeedForward(input_dim=2*len_b, output_dim=dm, hidden_dims=hidden_fourier, dropout=dropout_fourier)
        self.ff_raw = FeedForward(input_dim=2, output_dim=dp, hidden_dims=hidden_raw, dropout=dropout_raw)
        self.ff_premz = FeedForward(input_dim = 2*len_b+1, output_dim = self.sz, hidden_dims=hidden_premz, dropout=dropout_premz)


    def forward(self, mz:torch.Tensor, i:torch.Tensor, premz):

        # note: mz would be [batch, num_peaks, 1]

        # fill in raw encoding and pass through feedforward
        raw_encode = torch.stack([mz, i], dim=-1)  # [batch, num_peaks, 2]
        encoded_raw = self.ff_raw(raw_encode)  # [batch, num_peaks, dp]
        mzs = raw_encode[:, :, 0].unsqueeze(-1)  # why is pytorch so weird 

        b = self.b.unsqueeze(0).unsqueeze(0)

        # fill in fourier encoding
        sine_terms = torch.sin(mzs*b*2*torch.pi)
        cosine_terms = torch.cos(mzs*b*2*torch.pi)

        # initialize fourier array, fill, and run through feedforward
        fourier_encode = torch.empty(mz.shape[0], mz.shape[1], 2*self.len_b, device=mz.device)
        fourier_encode[:, :, ::2] = sine_terms
        fourier_encode[:, :, 1::2] = cosine_terms
        encoded_fourier = self.ff_fourier(fourier_encode)
        
        # append along correct axis
        x = torch.cat([encoded_fourier, encoded_raw], dim=-1)

        spectrum = x[:, 1:, :]
        precursor = x[:, 0, :].unsqueeze(1)

        return spectrum, precursor


# above architecture is more similar to literature, for testing for now
# below is kenny's code

# class SpecEmbed(nn.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         num_features = config.spec.num_features // 4
#         m_min = config.spec.m_min
#         m_max = config.spec.m_max

#         lows = num_features // 2
#         highs = num_features - lows - 1

#         b = torch.tensor(
#             [(1 / (i * ceil(m_max / lows))) for i in range(lows, 0, -1)] + [1] +
#             [(i * ceil(1 / m_min / highs)) for i in range(1, highs + 1, 1)]
#         )
#         self.b = b.unsqueeze(0)
#         self.b = nn.Parameter(self.b, requires_grad=False)
#         self.register_parameter("b", self.b)

#     def forward(self, mz: torch.Tensor, i: torch.Tensor):
#         mz = 2 * torch.pi * mz.unsqueeze(0).T @ self.b
#         i = 2 * torch.pi * i.unsqueeze(0).T @ self.b
#         return torch.cat([torch.sin(mz), torch.cos(mz), torch.sin(i), torch.cos(i)], dim=-1)
