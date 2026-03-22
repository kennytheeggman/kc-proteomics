import torch


class DatasetConfig:
    def __init__(self):
        self.training = "../datasets/IVE_v2_train.h5"
        self.eval = "../datasets/IVE_v2_val.h5"
        self.masking_prob = 0.1
        self.sigma = 0.01

class SpectrumConfig:
    def __init__(self):
        self.m_min = 1e-4
        self.m_max = 1e3
        self.num_features = 4096

        self.dm = 980
        self.dp = 44
        self.dropout_fourier = 0.5
        self.dropout_raw = 0.5
        self.dropout_premz = 0.5
        self.hidden_fourier = []
        self.hidden_raw = []
        self.hidden_premz = []

class LatentConfig:
    def __init__(self):
        self.encoder_num_layers = 8
        self.num_heads = 8

class SequenceConfig:
    def __init__(self):
        self.linear_num_layers = 4
        self.encoder_num_layers = 8
        self.decoder_num_layers = 8
        self.num_heads = 8
        self.max_seq_len = 60

class HyperConfig:
    def __init__(self):
        self.ctc_weight = 1.0
        self.mse_weight = 0.0
        self.max_length = 60
        self.d_model = 256
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 3e-5
        self.max_learning_rate = 1e-3
        self.checkpoint_name = "checkpoint.pth"


class Config:
    def __init__(self):
        # Constants
        self.AA_TYPES = 21
        self.SOS_TOKEN = 21
        self.EOS_TOKEN = 22
        self.PAD_TOKEN = 23

        self.aa_masses = torch.tensor([57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.04767, 160.030649, 113.084064, 113.084064, 114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485, 137.058912, 147.068414, 156.101111, 163.063329, 186.079313, 147.0354, 115.026943, 129.042594, 42.010565, 43.005814, 10000.0, 25.980265])
        self.tolerance = 5
        self.mass_res = int(5)

        self.cpu = "cpu"
        accelerator = torch.accelerator.current_accelerator()
        self.device = accelerator.type if accelerator else "cpu"

        # Hyperparameters
        self.data = DatasetConfig()
        self.spec = SpectrumConfig()
        self.embed = LatentConfig()
        self.seq = SequenceConfig()
        self.hyper = HyperConfig()

    def __str__(self):
        return f"Datasets(Training: \"{self.data.training}\", Eval: \"{self.data.eval}\")" 
    def __repr__(self):
        return str(self)


