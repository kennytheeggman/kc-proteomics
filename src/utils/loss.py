

import torch

from .ctc import encode


def get_loss(config):
    ctc_loss = torch.nn.CTCLoss(blank=0)
    mse_loss = torch.nn.MSELoss()

    def loss(prob_matrix: torch.Tensor, embed: torch.Tensor, decoded: torch.Tensor, sequence: str):
        loss1 = ctc_loss(prob_matrix, encode(sequence), torch.tensor([prob_matrix.shape[0]]), torch.tensor([len(sequence)]))
        loss2 = mse_loss(embed, decoded)
        return loss1 * config.hyper.ctc_weight + loss2 * config.hyper.mse_weight

    return loss
