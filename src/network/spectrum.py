# spectrum to vector and vector to spectrum modules
# !!! ignore all premz code i misunderstood data oops, it's also incomplete

from ..utils.config import Config
import torch.nn as nn
import torch
from ..network.linear import FeedForward

class SpectrumEmbed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # initialize variables for fourier embedding
        num_peaks = config.hyper.max_length
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
        self.b = torch.cat((torch.arange(m_max, 0, -1), m_min*torch.arange(k, 0, -1)))
        # self.register_buffer('b', self.b)

        # set up feed forward networks
        self.ff_fourier = FeedForward(input_dim=2*len_b, output_dim=dm, hidden_dims=hidden_fourier, dropout=dropout_fourier)
        self.ff_raw = FeedForward(input_dim=2, output_dim=dp, hidden_dims=hidden_raw, dropout=dropout_raw)
        self.ff_premz = FeedForward(input_dim = 2*len_b+1, output_dim = self.sz, hidden_dims=hidden_premz, dropout=dropout_premz)


    def forward(self, spectrum: tuple[torch.Tensor, torch.Tensor]):

        # note: mz would be [batch, num_peaks, 1]
        mz, i = spectrum
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

        return x[:, 1:, :], x[:, 0, :].unsqueeze(1)
