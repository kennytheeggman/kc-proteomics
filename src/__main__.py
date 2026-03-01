import argparse
import torch
from pathlib import Path
from .data.data import TrainingDataset


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


class Config:
    def __init__(self, dataset_training, dataset_eval):
        # Datasets
        self.dataset_training = Path(dataset_training)
        self.dataset_eval = Path(dataset_eval)

        # Torch Config
        self.cpu = "cpu"
        accelerator = torch.accelerator.current_accelerator()
        self.device = accelerator.type if accelerator else "cpu"

    def __str__(self):
        return f"Datasets(Training: \"{self.dataset_training.name}\", Eval: \"{self.dataset_eval.name}\")" 
    def __repr__(self):
        return str(self)


def args():
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--train', default='../datasets/IVE_v2_train.h5')
    parser.add_argument('--eval', default='../datasets/IVE_v2_val.h5')
    args = parser.parse_args()
    return Config(args.train, args.eval) 


if __name__ == "__main__":
    config = args()
    dataset = TrainingDataset(config)
    print(dataset[0])
    print(config.device)

