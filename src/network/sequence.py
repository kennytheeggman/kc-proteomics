

from torch import nn
import torch
import math

from src.network.linear import FeedForward
from src.utils.config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe
        x = x + pe[:, :x.size(1)]
        return self.dropout(x)

class SequenceEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.hyper.d_model, 
                config.seq.num_heads, 
                dim_feedforward=config.hyper.d_model,
                batch_first=True
            ),
            num_layers=config.seq.encoder_num_layers
        )
        self.positional_encoding = PositionalEncoding(config.hyper.d_model, config.hyper.max_length)
        self.ff1 = FeedForward(
            input_dim=1, 
            output_dim=config.hyper.d_model, 
            hidden_dims=[config.spec.dm+config.spec.dp]*5,
            bias=True
        )
        self.ff2 = FeedForward(
            input_dim=config.hyper.d_model, 
            output_dim=config.hyper.d_model, 
            hidden_dims=[config.spec.dm+config.spec.dp]*5, 
            bias=True
        )

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]):
        sequence, mask = data
        sequence = self.ff1.forward(sequence.unsqueeze(2))
        sequence = self.positional_encoding.forward(sequence)
        x = self.encoder.forward(sequence, src_key_padding_mask=mask)
        x = self.ff2.forward(x)
        return x


class SequenceDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.hyper.d_model, 
                config.seq.num_heads, 
                dim_feedforward=config.hyper.d_model,
                batch_first=True
            ),
            num_layers=config.seq.decoder_num_layers
        )
        self.ff = FeedForward(
            input_dim=config.hyper.d_model, 
            output_dim=config.AA_TYPES + 3, 
            hidden_dims=[], 
            bias=True
        )

    def forward(self, data: tuple[torch.Tensor, torch.Tensor], embedding: torch.Tensor):
        mask = torch.triu(torch.ones(self.config.hyper.max_length, self.config.hyper.max_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        sequence, padding_mask = data
        x = self.decoder.forward(sequence, embedding, tgt_mask=mask.to(self.config.device), tgt_key_padding_mask=padding_mask)
        x = self.ff.forward(x)
        return x
