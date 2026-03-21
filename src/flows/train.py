

import torch
from torch.utils.data import DataLoader

from src.data.data import Peptide
from src.network.network import Model, decode, encode
from src.utils.config import Config


def run(config: Config, model: Model, loss_fn, optimizer, scheduler, train_dataloader: DataLoader[Peptide], eval_dataloader: DataLoader[Peptide]):
    # for epoch in range(config.hyper.epochs):
        for idx, batch in enumerate(train_dataloader):
            mass, mz, i, peptide = batch
            optimizer.zero_grad()
            logits = model.forward((mz, i), peptide)
            loss: torch.Tensor = loss_fn(logits, peptide)
            loss.backward()
            optimizer.step()
            print(f"Step {idx} loss: {loss.item()} pred: {decode(logits, config)[0]}, target: {peptide[0]}")
            # scheduler.step()
        # print(f"Epoch {epoch} loss: {loss.item()}")
