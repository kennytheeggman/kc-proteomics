

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from src.data.data import Peptide, EvalDataset
from src.network.network import Model, decode, encode
from src.utils.config import Config


def run(config: Config, loss_fn, eval_dataset: EvalDataset):
    model = Model(config)
    model.load_state_dict(torch.load(config.hyper.checkpoint_name, map_location=config.device))
    model.to(config.device)
    idx = int(input("Idx: "))
    sample = eval_dataset[idx]
    encoded = encode(["YGTCIYQR"], config)
    with torch.no_grad():
        mz = sample.mz.unsqueeze(0).to(config.device)
        i = sample.i.unsqueeze(0).to(config.device)
        sequence = ["YGTCIYQR"]
        logits = model((mz, i), sequence)
        # loss = loss_fn(logits, sequence)
    prediction = decode(logits, config)[0]
    print(sample.sequence)
    print(prediction)
    print(encoded[0])
    print(torch.argmax(logits, dim=-1))
