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
    # run(config, loss_fn, EvalDataset(config))

