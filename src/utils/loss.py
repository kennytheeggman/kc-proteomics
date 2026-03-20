import torch
import torch.nn as nn

from .ctc import encode


def get_loss(config):
    ce_loss = nn.CrossEntropyLoss(
        ignore_index=config.PAD,
        label_smoothing=config.hyper.label_smoothing
    )
    mse_loss = nn.MSELoss()

    def loss(logits: torch.Tensor, embed: torch.Tensor, decoded: torch.Tensor, tgt: torch.Tensor, tgt_padding_mask: torch.Tensor):
        batch_size, seq_len, vocab_size = logits.shape
        
        logits_flat = logits.view(-1, vocab_size)
        tgt_flat = tgt.view(-1)
        
        loss_ce = ce_loss(logits_flat, tgt_flat)
        loss_mse = mse_loss(embed, decoded)

        return loss_ce # + loss_mse * config.hyper.mse_weight

    return loss
