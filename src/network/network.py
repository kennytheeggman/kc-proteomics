

from torch import nn
import torch

from src.network.latent import LatentEncoder
from src.network.sequence import SequenceDecoder, SequenceEncoder
from src.network.spectrum import SpectrumEmbed
from src.utils.config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encode_spectrum = SpectrumEmbed(config)
        self.encode_embedding = LatentEncoder(config)
        self.encode_sequence = SequenceEncoder(config)
        self.decode_sequence = SequenceDecoder(config)

    def forward(self, spectrum: tuple[torch.Tensor, torch.Tensor], sequence: list[str]):
        spec, prec = self.encode_spectrum.forward(spectrum)
        embedding = self.encode_embedding.forward(spec, prec)
        encoded, mask = encode(sequence, self.config)
        target = self.encode_sequence.forward((encoded.to("cuda"), mask.to("cuda")))
        logits = self.decode_sequence.forward((target, mask.to("cuda")), embedding)
        # return logits
        return nn.functional.log_softmax(logits, dim=2)

def encode(sequence: list[str], config: Config):
    ENCODE_MAP = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'm': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    result: torch.Tensor = torch.empty(0, config.hyper.max_length)
    masks: torch.Tensor = torch.empty(0, config.hyper.max_length, dtype=torch.bool)
    for seq in sequence:
        encoded = torch.tensor([config.SOS_TOKEN] + [ENCODE_MAP[aa] for aa in seq] + [config.EOS_TOKEN])
        if (len(encoded) > config.hyper.max_length):
            encoded = encoded[:config.hyper.max_length]
        elif (len(encoded) < config.hyper.max_length):
            encoded = nn.functional.pad(encoded, (0, config.hyper.max_length - len(encoded)), value=config.PAD_TOKEN)
        emask = torch.tensor([False] * len(encoded) + [True] * (config.hyper.max_length - len(encoded)))
        result = torch.cat([result, encoded.unsqueeze(0)], dim=0)
        masks = torch.cat([masks, emask.unsqueeze(0)], dim=0)
    return result, masks

def decode(logits: torch.Tensor, config: Config):
    DECODE_MAP = [ 
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'm', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ]
    sequences: list[str] = []
    for log in logits:
        decoded = torch.argmax(log, dim=-1)
        sequence: list[str] = []
        for aa in decoded:
            if aa == config.SOS_TOKEN:
                continue
            elif aa == config.EOS_TOKEN or aa == config.PAD_TOKEN:
                break
            else:
                sequence.append(DECODE_MAP[aa])
        sequences.append("".join(sequence))
    return sequences
