import argparse

import torch

from src.network.network import Model
from src.utils.ctc import decode, reduce 

from .data.data import TrainingDataset
from .utils.config import Config


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


def args():
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--train', default='../datasets/IVE_v2_train.h5')
    parser.add_argument('--eval', default='../datasets/IVE_v2_val.h5')
    args = parser.parse_args()
    return Config(args.train, args.eval) 


if __name__ == "__main__":
    config = args()
    dataset = TrainingDataset(config)
    model = Model(config)
    model.to(config.device)
    print(dataset[0])
    charge, premz, mz, i, peptide = dataset[0]
    prob_matrix, encoded, decoded = model(mz.to(config.device), i.to(config.device))
    print(prob_matrix.to(config.cpu), prob_matrix.shape)
    sequence = decode(prob_matrix, log=True)
    processed = reduce(sequence)

    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
    loss = ctc_loss(prob_matrix.to(config.cpu), sequence.to(config.cpu), torch.tensor([prob_matrix.shape[0]]), torch.tensor([sequence.shape[0]]))
    print("".join(processed))
    print((decoded - encoded).to(config.cpu), decoded.shape)
    print(loss)
    loss.backward()
