

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from src.data.data import Peptide
from src.network.network import Model, decode, encode
from src.utils.config import Config


def run(config: Config, model: Model, loss_fn, optimizer, scheduler, train_dataloader: DataLoader[Peptide], eval_dataloader: DataLoader[Peptide]):
    # for epoch in range(config.hyper.epochs):
    writer = SummaryWriter(log_dir="runs/run1")
    global_step = 0
    moving_avg = None
    weight = 0.1
    for idx, batch in enumerate(train_dataloader):
        mass, mz, i, peptide = batch
        optimizer.zero_grad()
        logits = model.forward((mz.to(config.device), i.to(config.device)), peptide)
        loss: torch.Tensor = loss_fn(logits, peptide)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"Step {idx} loss: {loss.item()} pred: {decode(logits, config)[0]}, target: {peptide[0]}")
        if idx % 1000 == 0:
            torch.save(model.state_dict(), config.hyper.checkpoint_name)
        # scheduler.step()
        if moving_avg is None:
            moving_avg = loss.item()
        else:
            moving_avg = loss.item() * weight + (1 - weight) * moving_avg
        writer.add_scalar("Loss/train", loss.item(), global_step)
        # writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("Moving Average Loss", moving_avg, global_step)
        global_step += 1
    # print(f"Epoch {epoch} loss: {loss.item()}")
