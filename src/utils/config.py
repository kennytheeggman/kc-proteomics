from pathlib import Path
import torch


class DatasetConfig:
    def __init__(self):
        self.training = "../../datasets/IVE_v2/IVE_v2_train.h5"
        self.eval = "../../datasets/IVE_v2/IVE_v2_val.h5"
        self.masking_prob = 0.1
        self.sigma = 0.01

class SpectrumConfig:
    def __init__(self):
        self.m_min = 1e-4
        self.m_max = 1e3
        self.num_features = 4096

class LatentConfig:
    def __init__(self):
        self.d_model = 288
        self.num_layers = 4
        self.linear_num_layers = 4
        self.encoder_num_layers = 4
        self.decoder_num_layers = 4

class SequenceConfig:
    def __init__(self):
        self.linear_num_layers = 4
        self.encoder_num_layers = 4

class HyperConfig:
    def __init__(self):
        self.ctc_weight = 1.0
        self.mse_weight = 0.0
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 1e-4
        self.checkpoint_name = "checkpoint.pth"


class Config:
    def __init__(self):
        # Constants
        self.AA_TYPES = 27

        # decoding parameters
        self.aa_masses = torch.tensor([57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.04767, 160.030649, 113.084064, 113.084064, 114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485, 137.058912, 147.068414, 156.101111, 163.063329, 186.079313, 147.0354, 115.026943, 129.042594, 42.010565, 43.005814, 10000.0, 25.980265])
        self.tolerance = 10E-3
        self.mass_res = int(100)

        # Torch Config
        self.cpu = "cpu"
        accelerator = torch.accelerator.current_accelerator()
        self.device = accelerator.type if accelerator else "cpu"
        self.device = "cpu"

        # Hyperparameters
        self.num_peaks = 48
        self.data = DatasetConfig()
        self.spec = SpectrumConfig()
        self.embed = LatentConfig()
        self.seq = SequenceConfig()
        self.hyper = HyperConfig()

    def __str__(self):
        return f"Datasets(Training: \"{self.data.training}\", Eval: \"{self.data.eval}\")" 
    def __repr__(self):
        return str(self)


