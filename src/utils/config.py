from pathlib import Path
import torch


class SpectrumConfig:
    def __init__(self):
        self.m_min = 1e-4
        self.m_max = 1e3
        self.num_features = 4096

class HyperConfig:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 1e-4


class Config:
    def __init__(self, dataset_training, dataset_eval):
        # Datasets
        self.dataset_training = Path(dataset_training)
        self.dataset_eval = Path(dataset_eval)

        # Torch Config
        self.cpu = "cpu"
        accelerator = torch.accelerator.current_accelerator()
        self.device = accelerator.type if accelerator else "cpu"

        # Hyperparameters
        self.spec = SpectrumConfig()
        self.hyper = HyperConfig()

    def __str__(self):
        return f"Datasets(Training: \"{self.dataset_training.name}\", Eval: \"{self.dataset_eval.name}\")" 
    def __repr__(self):
        return str(self)


