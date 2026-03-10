# latent space to sequence and sequence to latent space (optional) modules


from math import ceil
from torch import nn
import torch
from ..utils.config import Config
from ..network.linear import FeedForward

class EmbedSequence(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.embed.d_model
        self.d_output = config.AA_TYPES
        self.num_heads = config.seq.num_heads
        self.linear_layers = config.seq.linear_num_layers
        self.encoder_layers = config.seq.encoder_num_layers

        self.embedding_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.embed.d_model, self.num_heads, batch_first=True),
            self.encoder_layers
        )

        # linear_layers = [
        #     nn.Linear(
        #         ceil((self.d_output - self.d_model) * (i / self.linear_layers) + self.d_model), 
        #         ceil((self.d_output - self.d_model) * ((i + 1) / self.linear_layers) + self.d_model)
        #     ) for i in range(self.linear_layers)
        # ]
        # self.vector_stack = nn.Sequential(*linear_layers)

        hidden_dims = [ceil((self.d_output - self.d_model) * (i / self.linear_layers) + self.d_model) for i in range(1, self.linear_layers)]
        self.vector_stack = FeedForward(input_dim=self.d_model, output_dim=self.d_output, hidden_dims=hidden_dims)

    def forward(self, embedding):
        raw_values = self.embedding_stack(embedding)
        raw_matrix = self.vector_stack(raw_values)
        prob_matrix = torch.log_softmax(raw_matrix, dim=-1)
        return prob_matrix
