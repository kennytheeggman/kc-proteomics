# vector to latent space and latent space to vector modules


from torch import nn
from math import ceil

from ..utils.config import Config


class EmbedEncode(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_input = config.spec.num_features
        self.d_model = config.embed.d_model
        self.num_peaks = config.num_peaks
        self.layers = config.embed.linear_num_layers
        self.config = config

        # linear layers to project to model dimension, number of features in each layer is lerped 
        linear_layers = [
            nn.Linear(
               ceil((self.d_model - self.d_input) * (i / self.layers) + self.d_input), 
               ceil((self.d_model - self.d_input) * ((i + 1) / self.layers) + self.d_input)
            ) for i in range(self.layers)
        ]
        self.vector_stack = nn.Sequential(*linear_layers)

        self.spectrum_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.embed.d_model, self.num_peaks),
            config.embed.encoder_num_layers
        )
        self.precursor_stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(config.embed.d_model, self.num_peaks),
            config.embed.decoder_num_layers
        )

    def forward(self, spectrum, precursor):
        projected_spectrum = self.vector_stack(spectrum)
        projected_precursor = self.vector_stack(precursor).unsqueeze(0).repeat(self.num_peaks, 1)
        transformed_spectrum = self.spectrum_stack(projected_spectrum)
        transformed_combined = self.precursor_stack(transformed_spectrum, projected_precursor)
        return transformed_combined


# inverse structure of the above
class EmbedDecode(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.embed.d_model
        self.d_output = config.spec.num_features
        self.num_peaks = config.num_peaks
        self.layers = config.embed.linear_num_layers
        self.config = config

        self.spectrum_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.embed.d_model, self.num_peaks),
            config.embed.encoder_num_layers + config.embed.decoder_num_layers
        )

        # linear layers to project to model dimension, number of features in each layer is lerped 
        linear_layers = [
            nn.Linear(
                ceil((self.d_output - self.d_model) * (i / self.layers) + self.d_model), 
                ceil((self.d_output - self.d_model) * ((i + 1) / self.layers) + self.d_model)
            ) for i in range(self.layers)
        ]
        self.vector_stack = nn.Sequential(*linear_layers)

    def forward(self, embedding):
        transformed_spectrum = self.spectrum_stack(embedding)
        predicted_spectrum = self.vector_stack(transformed_spectrum)
        return predicted_spectrum
