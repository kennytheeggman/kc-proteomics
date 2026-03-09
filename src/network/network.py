# composition of modules into a) spectrum to latent to sequence and b) latent to spectrum


from torch import nn

from ..network.latent import EmbedDecode, EmbedEncode
from ..network.sequence import EmbedSequence
from ..network.spectrum import SpecEmbed
from ..utils.config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.se = SpecEmbed(config)
        self.ee = EmbedEncode(config)
        self.es = EmbedSequence(config)
        self.ed = EmbedDecode(config)

    def forward(self, mz, i, premz):
        seq_embed, p_embed = self.se.forward(mz, i, premz)
        encoded = self.ee.forward(seq_embed, p_embed)  # seq_embed shape is (seq_len, d_model) and p_embed shape is (d_model)
        prob_matrix = self.es.forward(encoded)
        decoded = self.ed.forward(encoded)
        return prob_matrix, seq_embed, decoded
