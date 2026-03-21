

from torch import nn
import torch

from src.network.linear import FeedForward
from src.utils.config import Config


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
        x = self.decoder.forward(sequence, embedding, tgt_mask=mask, tgt_key_padding_mask=padding_mask)
        x = self.ff.forward(x)
        return x
