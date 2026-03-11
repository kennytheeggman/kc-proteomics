

import torch

from .ctc import encode

def get_loss(config):
    ctc_loss = torch.nn.CTCLoss(blank=0)
    mse_loss = torch.nn.MSELoss()

    def loss(prob_matrix: torch.Tensor, embed: torch.Tensor, decoded: torch.Tensor, sequences):

        # ctc loss is NOT batch first
        prob_matrix_ctc = prob_matrix.permute(1, 0, 2)

        batch_sz = prob_matrix_ctc.shape[1]
        seq_len = prob_matrix_ctc.shape[0]

        # set up things for ctc
        encoded_sequences = torch.cat([encode(s.decode()) for s in sequences])  # flat 1d tensor
        input_lengths = torch.full((batch_sz,), seq_len, dtype=torch.long)  # lengths of each spectrum seq (where to slice encoded_sequences)
        target_lengths = torch.tensor([len(s.decode()) for s in sequences], dtype=torch.long)  # lengths of each peptide seq (where to slice encoded peptide sequences)

        loss1 = ctc_loss(prob_matrix_ctc, encoded_sequences, input_lengths, target_lengths)
        loss2 = mse_loss(embed, decoded)

        return loss1 * config.hyper.ctc_weight + loss2 * config.hyper.mse_weight

    return loss
