import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class PeptideDataset(Dataset):

    def __init__(self, file_path):
        super().__init__();
        file = h5py.File(file_path, 'r')
        obj = file['charges']
        assert isinstance(obj, h5py.Dataset)
        self.charges: h5py.Dataset = obj 
        obj = file['peptides']
        assert isinstance(obj, h5py.Dataset)
        self.peptides: h5py.Dataset = obj 
        obj = file['premzs']
        assert isinstance(obj, h5py.Dataset)
        self.premzs: h5py.Dataset = obj 
        obj = file['spectra']
        assert isinstance(obj, h5py.Dataset)
        self.spectra: h5py.Dataset = obj

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
