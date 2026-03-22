import argparse
import torch
from torch.utils.data import DataLoader

from src.flows.train import run
from src.network.network import Model
from src.utils.loss import get_loss

from .data.data import EvalDataset, TrainingDataset, collate_fn
from .utils.config import Config


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


def args():
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    return Config() 


if __name__ == "__main__":
    config = args()
    dataset = TrainingDataset(config)
    model = Model(config)
    loss_fn = get_loss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper.learning_rate)
    train_dataloader = DataLoader(dataset, batch_size=config.hyper.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(EvalDataset(config), batch_size=config.hyper.batch_size, shuffle=False, collate_fn=collate_fn)
    run(config, model.to("cuda"), loss_fn, optimizer, None, train_dataloader, eval_dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hyper.epochs * len(train_dataloader),  eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.hyper.max_learning_rate, steps_per_epoch=len(train_dataloader), epochs=config.hyper.epochs)
