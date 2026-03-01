import argparse

import torch

from src.network.network import Model 

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
    mz, i = torch.randn(16), torch.randn(16)
    prob_matrix, decoded = model(mz, i)
    print(prob_matrix, prob_matrix.shape)
    print(decoded, decoded.shape)
