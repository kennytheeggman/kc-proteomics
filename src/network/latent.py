

from torch import nn

from src.network.linear import FeedForward
from src.utils.config import Config


class LatentEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(config.hyper.d_model, config.embed.num_heads, dim_feedforward=config.hyper.d_model),
            num_layers=config.embed.encoder_num_layers
        )
        self.ff1 = FeedForward(input_dim=config.spec.dm+config.spec.dp, output_dim=config.hyper.d_model, hidden_dims=[config.spec.dm+config.spec.dp]*5, bias=True)
        self.ff2 = FeedForward(input_dim=config.hyper.d_model, output_dim=config.hyper.d_model, hidden_dims=[config.spec.dm+config.spec.dp]*5, bias=True)

    def forward(self, spectrum, precursor):
        spectrum = self.ff1.forward(spectrum)
        precursor = self.ff1.forward(precursor)
        x = self.encoder.forward(spectrum, precursor.repeat(1, self.config.hyper.max_length, 1))  # why need to repeat?
        x = self.ff2.forward(x)
        return x
