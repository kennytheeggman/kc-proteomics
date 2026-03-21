import torch

from src.network.network import encode
from src.utils.config import Config

def get_loss(config: Config):
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN)
    def loss_fn(logits: torch.Tensor, targets: list[str]):
        tgt, _ = encode(targets, config)
        padding = torch.tensor([config.PAD_TOKEN]).unsqueeze(1).repeat(config.hyper.batch_size, 1)
        tgt = torch.cat([tgt[:, 1:], padding], dim=1)
        logits = logits.permute(0, 2, 1)
        tgt = tgt.type(dtype=torch.long)
        # print(f"Input: {encode(targets, config)[0][0]}, Target: {tgt[0]}")
        return ce_loss.forward(logits, tgt)
    return loss_fn
