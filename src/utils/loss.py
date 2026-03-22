import torch
import torch.nn as nn

from src.network.network import encode
from src.utils.config import Config

def get_loss(config: Config):
    ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # ce_loss = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN, label_smoothing=0.1, weight=torch.tensor([1.0]*21+[1.0]*3).to("cuda"))
    def loss_fn(logits: torch.Tensor, targets: list[str]):
        tgt, _ = encode(targets, config)
        padding = torch.tensor([config.PAD_TOKEN]).unsqueeze(1).repeat(config.hyper.batch_size, 1)
        tgt = torch.cat([tgt[:, 1:], padding], dim=1)
        logits = logits.permute(0, 2, 1).to("cuda")
        tgt = tgt.type(dtype=torch.long).to("cuda")
        # print(logits.shape, tgt.shape)
        return ce_loss.forward(logits, tgt)
    return loss_fn
