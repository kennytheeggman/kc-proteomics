import argparse

from .network.network import Model
from .utils.loss import get_loss 

from .data.data import TrainingDataset
from .utils.config import Config


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


def args():
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--train', default='../datasets/IVE_v2_train.h5')
    parser.add_argument('--eval', default='../datasets/IVE_v2_val.h5')
    args = parser.parse_args()
    return Config() 


if __name__ == "__main__":
    config = args()
    dataset = TrainingDataset(config)
    model = Model(config)
    model.to(config.device)
    loss_fn = get_loss(config)
    print(dataset[0])
    charge, premz, mz, i, peptide = dataset[0]
    peptide = peptide.decode()
    prob_matrix, encoded, decoded = model(mz.to(config.device), i.to(config.device))
    loss = loss_fn(prob_matrix.to(config.cpu), encoded.to(config.cpu), decoded.to(config.cpu), peptide)
    print(loss)
