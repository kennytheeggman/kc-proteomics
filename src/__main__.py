import argparse
import torch
from pathlib import Path
from .data.data import TrainingDataset


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


class Config:
    def __init__(self, dataset_training, dataset_eval):
        self.dataset_training = Path(dataset_training)
        self.dataset_eval = Path(dataset_eval)
    def __str__(self):
        return f"Datasets(Training: \"{self.dataset_training.name}\", Eval: \"{self.dataset_eval.name}\")" 
    def __repr__(self):
        return str(self)


def args():
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('train')
    parser.add_argument('eval')
    args = parser.parse_args()
    return Config(args.train, args.eval) 


if __name__ == "__main__":
    config = args()
    dataset = TrainingDataset(config)
    print(dataset[0])
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(device)

