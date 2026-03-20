from torch import nn

from ..network.latent import EmbedDecode, EmbedEncode
from ..network.sequence import EmbedSequence
from ..network.spectrum import SpecEmbed
from ..utils.config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.se = SpecEmbed(config)
        self.ee = EmbedEncode(config)
        self.es = EmbedSequence(config)
        self.ed = EmbedDecode(config)

    def forward(self, mz, i, premz, tgt=None, tgt_key_padding_mask=None):
        seq_embed, p_embed = self.se.forward(mz, i, premz)
        encoded = self.ee.forward(seq_embed, p_embed)
        
        if tgt is not None:
            logits = self.es.forward(encoded, tgt, tgt_key_padding_mask)
            decoded = self.ed.forward(encoded)
            return logits, seq_embed, decoded
        
        generated = self.es.forward(encoded)
        decoded = self.ed.forward(encoded)
        return generated, seq_embed, decoded

    def encode(self, mz, i, premz):
        seq_embed, p_embed = self.se.forward(mz, i, premz)
        encoded = self.ee.forward(seq_embed, p_embed)
        return encoded

    def decode_sequence(self, memory, tgt=None, tgt_key_padding_mask=None):
        if tgt is not None:
            return self.es.forward(memory, tgt, tgt_key_padding_mask)
        return self.es.forward(memory)
