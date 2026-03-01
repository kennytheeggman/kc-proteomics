import argparse
from .data.data import TrainingDataset
from .utils.config import Config


PROG_NAME = 'KC Proteomics'
PROG_DESC = 'Training and inference and fine-tuning for semi-supervised proteomics models'


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

