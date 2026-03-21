from typing import Iterable
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

from src.utils.config import Config


class Peptide:
    def __init__(self, charge: torch.Tensor, mass: torch.Tensor, spectrum: tuple[torch.Tensor, torch.Tensor], sequence: bytes):
        self.precursor_charge = charge
        self.precursor_mass = mass
        self.mz, self.i = spectrum
        self.sequence = sequence

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

    def __getitem__(self, idx) -> Peptide:
        spectrum_size = len(self.spectra[idx])
        mz, i = (
            torch.from_numpy(np.array(self.spectra[idx])[:int(spectrum_size/2)]),
            torch.from_numpy(np.array(self.spectra[idx])[int(spectrum_size/2):])
        )
        return Peptide(
            torch.tensor(self.charges[idx]).int(),
            torch.tensor(self.premzs[idx]).float(),
            (mz, i), self.peptides[idx].decode()
        )

class TrainingDataset(PeptideDataset):
    def __init__(self, config: Config):
        super().__init__(config.data.training, config.hyper.max_length)
        self.masking_prob = config.data.masking_prob
        self.sigma = config.data.sigma

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # # select peaks to remove
        mask = torch.rand(data.mz.shape) < self.masking_prob
        mz = data.mz[~mask]
        i = data.i[~mask] 
        # change all peaks intensities by small amount
        mz = mz * (1 + (torch.randn(mz.shape) - 0.5) * self.sigma)
        i = i * (1 + (torch.randn(i.shape) - 0.5) * self.sigma)
        # truncate or pad to num_peaks
        # padding
        if (len(mz) < self.num_peaks):
            mz = torch.cat([mz, torch.zeros(self.num_peaks - mz.shape[0])])
            i = torch.cat([i, torch.zeros(self.num_peaks - i.shape[0])])
        elif (len(mz) > self.num_peaks):
            mz, i = mz[:self.num_peaks], i[:self.num_peaks]
        mz = torch.cat([data.precursor_charge, mz])
        i = torch.cat([torch.tensor(1.1).unsqueeze(0), i])
        return Peptide(data.precursor_charge, data.precursor_mass, (mz, i), data.sequence)

class EvalDataset(PeptideDataset):
    def __init__(self, config: Config):
        super().__init__(config.data.eval, config.hyper.max_length)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # truncate or pad to num_peaks
        mz = data.mz
        i = data.i
        if (len(mz) < self.num_peaks):
            mz = torch.cat([mz, torch.zeros(self.num_peaks - mz.shape[0])])
            i = torch.cat([i, torch.zeros(self.num_peaks - i.shape[0])])
        elif (len(mz) > self.num_peaks):
            mz, i = mz[:self.num_peaks], i[:self.num_peaks]
        mz = torch.cat([data.precursor_mass, mz])
        i = torch.cat([torch.tensor(1.1).unsqueeze(0), i])
        return Peptide(data.precursor_charge, data.precursor_mass, (mz, i), data.sequence)
    

def collate_fn(batch: Iterable[Peptide]):
    prem = [b.precursor_mass for b in batch]
    mz = [b.mz for b in batch]
    i = [b.i for b in batch]
    peptide = [b.sequence for b in batch]
    # tgt_padding_mask = [b.tgt_padding_mask for b in batch]
    return torch.stack(prem), torch.stack(mz), torch.stack(i), peptide
