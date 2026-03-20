import argparse

import torch
from torch.utils.data import DataLoader

from src.flows.train import run
from src.utils.ctc import encode, reduce

from .network.network import Model
from .utils.loss import get_loss 

from .data.data import EvalDataset, TrainingDataset, autoregressive_collate_fn
from .utils.config import Config


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


def args():
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--train', default='../datasets/IVE_v2_train.h5')
    parser.add_argument('--eval', default='../datasets/IVE_v2_val.h5')
    args = parser.parse_args()
    return Config() 


def get_collate_fn(config):
    def collate(batch):
        return autoregressive_collate_fn(batch, config)
    return collate


if __name__ == "__main__":
    config = args()
    dataset = TrainingDataset(config)
    model = Model(config)
    model.to(config.device)
    loss_fn = get_loss(config)

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.hyper.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(config)
    )
    eval_dataloader = DataLoader(
        EvalDataset(config),
        batch_size=config.hyper.batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(config)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.hyper.max_learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=config.hyper.epochs
    )
    run(config, model, loss_fn, optimizer, scheduler, train_dataloader, eval_dataloader)
