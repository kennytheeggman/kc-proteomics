import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class PeptideDataset(Dataset):

    def __init__(self, file_path):
        super().__init__();
        file = h5py.File(file_path, 'r')
        self.charges = file['charges']
        self.mapping = file['mapping']
        self.peptides = file['peptides']
        self.premzs = file['premzs']
        self.spectra = file['spectra']

    def __getitem__(self, idx):
        spectrum_size = len(self.spectra[idx])
        return (
            torch.tensor(self.charges[idx]).int(),
            self.peptides[idx],
            torch.tensor(self.premzs[idx]).float(),
            torch.from_numpy(np.array(self.spectra[idx])[:int(spectrum_size/2)]),
            torch.from_numpy(np.array(self.spectra[idx])[int(spectrum_size/2):])
        )

class TrainingDataset(PeptideDataset):
    def __init__(self, config):
        super().__init__(config.dataset_training)

class EvalDataset(PeptideDataset):
    def __init__(self, config):
        super().__init__(config.dataset_eval)
