import argparse

import torch
from torch.utils.data import DataLoader

from src.flows.train import run

from .network.network import Model
from .utils.loss import get_loss 

from .data.data import EvalDataset, TrainingDataset
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

    train_dataloader = DataLoader(dataset, batch_size=config.hyper.batch_size, shuffle=True)
    eval_dataloader = DataLoader(EvalDataset(config), batch_size=config.hyper.batch_size, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.hyper.learning_rate)
    run(config, model, loss_fn, optimizer, train_dataloader, eval_dataloader)
