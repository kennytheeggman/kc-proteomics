# composition of modules into a) spectrum to latent to sequence and b) latent to spectrum


from torch import nn

from src.network.latent import EmbedDecode, EmbedEncode
from src.network.sequence import EmbedSequence
from src.network.spectrum import SpecEmbed
from src.utils.config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.se = SpecEmbed(config)
        self.ee = EmbedEncode(config)
        self.es = EmbedSequence(config)
        self.ed = EmbedDecode(config)

    def forward(self, mz, i):
        embed = self.se.forward(mz, i)
        encoded = self.ee.forward(embed, embed[0])
        prob_matrix = self.es.forward(encoded)
        decoded = self.ed.forward(encoded)
        return prob_matrix, embed, decoded
