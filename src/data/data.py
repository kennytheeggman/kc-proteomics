import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class PeptideDataset(Dataset):

    def __init__(self, file_path, num_peaks):
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

        self.num_peaks = num_peaks

    def __len__(self):
        return len(self.charges)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bytes]:
        spectrum_size = len(self.spectra[idx])
        mz, i = (
            torch.from_numpy(np.array(self.spectra[idx])[:int(spectrum_size/2)]),
            torch.from_numpy(np.array(self.spectra[idx])[int(spectrum_size/2):])
        )
        return (
            torch.tensor(self.charges[idx]).int(),
            torch.tensor(self.premzs[idx]).float(),
            mz, i, self.peptides[idx]
        )

class TrainingDataset(PeptideDataset):
    def __init__(self, config):
        super().__init__(config.data.training, config.num_peaks)
        self.masking_prob = config.data.masking_prob
        self.sigma = config.data.sigma

    def __getitem__(self, idx):
        charge, premz, mz, i, peptide = super().__getitem__(idx)

        # temporarily remove everything

        # # select peaks to remove
        # mask = torch.rand(mz.shape) < self.masking_prob
        # mz = mz[~mask]
        # i = i[~mask] 
        # # change all peaks intensities by small amount
        # mz = mz * (1 + (torch.randn(mz.shape) - 0.5) * self.sigma)
        # i = i * (1 + (torch.randn(i.shape) - 0.5) * self.sigma)
        # # truncate or pad to num_peaks
        mz = torch.cat([premz, mz])
        i = torch.cat([torch.tensor(1.1).unsqueeze(0), i])
        # padding
        if (len(mz) < self.num_peaks):
            mz = torch.cat([mz, torch.zeros(self.num_peaks - mz.shape[0])])
            i = torch.cat([i, torch.zeros(self.num_peaks - i.shape[0])])
        elif (len(mz) > self.num_peaks):
            mz, i = mz[:self.num_peaks], i[:self.num_peaks]
        return charge, premz, mz, i, peptide

class EvalDataset(PeptideDataset):
    def __init__(self, config):
        super().__init__(config.data.eval, config.num_peaks)
    
    def __getitem__(self, idx):
        charge, premz, mz, i, peptide = super().__getitem__(idx)
        # truncate or pad to num_peaks
        mz = torch.cat([premz, mz])
        i = torch.cat([torch.tensor(1.1).unsqueeze(0), i])
        if (len(mz) < self.num_peaks):
            mz = torch.cat([mz, torch.zeros(self.num_peaks - mz.shape[0])])
            i = torch.cat([i, torch.zeros(self.num_peaks - i.shape[0])])
        elif (len(mz) > self.num_peaks):
            mz, i = mz[:self.num_peaks], i[:self.num_peaks]
        return charge, premz, mz, i, peptide
    

def collate_fn(batch):
    charge, premz, mz, i, peptide = zip(*batch)
    return torch.stack(charge), torch.stack(premz), torch.stack(mz), torch.stack(i), torch.stack(peptide)