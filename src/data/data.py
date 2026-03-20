import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class PeptideDataset(Dataset):

    def __init__(self, file_path, num_peaks):
        super().__init__()
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

        mask = torch.rand(mz.shape) < self.masking_prob
        mz = mz[~mask]
        i = i[~mask] 
        mz = mz * (1 + (torch.randn(mz.shape) - 0.5) * self.sigma)
        i = i * (1 + (torch.randn(i.shape) - 0.5) * self.sigma)
        mz = torch.cat([premz, mz])
        i = torch.cat([torch.tensor(1.1).unsqueeze(0), i])
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
    return torch.stack(charge), torch.stack(premz), torch.stack(mz), torch.stack(i), list(peptide)


def autoregressive_collate_fn(batch, config):
    charge, premz, mz, i, peptides = zip(*batch)
    
    max_len = config.seq.max_seq_len
    
    src_seqs = []
    tgt_seqs = []
    tgt_padding_mask = []
    
    for peptide in peptides:
        seq_str = peptide.decode()
        tgt = encode_autoregressive(seq_str, config)
        tgt_len = len(tgt)
        
        src_seq = [config.SOS] + tgt[:-1]
        tgt_seq = tgt
        
        pad_len = max_len - tgt_len
        if pad_len > 0:
            src_seq = src_seq + [config.PAD] * pad_len
            tgt_seq = tgt_seq + [config.PAD] * pad_len
        else:
            src_seq = src_seq[:max_len]
            tgt_seq = tgt_seq[:max_len]
        
        src_seqs.append(src_seq)
        tgt_seqs.append(tgt_seq)
        tgt_padding_mask.append([False] * tgt_len + [True] * (max_len - tgt_len))
    
    return (
        torch.stack(charge),
        torch.stack(premz),
        torch.stack(mz),
        torch.stack(i),
        torch.tensor(src_seqs, dtype=torch.long),
        torch.tensor(tgt_seqs, dtype=torch.long),
        torch.tensor(tgt_padding_mask, dtype=torch.bool)
    )


def encode_autoregressive(sequence: str, config) -> list:
    ENCODE_MAP = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
        'I': 7, 'K': 8, 'L': 9, 'M': 10, 'm': 11, 'N': 12, 'P': 13,
        'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    }
    encoded = [config.SOS]
    for aa in sequence:
        if aa in ENCODE_MAP:
            encoded.append(ENCODE_MAP[aa])
    encoded.append(config.EOS)
    return encoded


def decode_autoregressive(sequence: torch.Tensor, config) -> str:
    DECODE_MAP = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'm', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ]
    result = []
    for idx in sequence:
        if isinstance(idx, torch.Tensor):
            idx_val = idx.item()
        else:
            idx_val = idx
        if idx_val == config.SOS:
            continue
        if idx_val == config.EOS or idx_val == config.PAD:
            break
        if 0 <= idx_val < len(DECODE_MAP):
            result.append(DECODE_MAP[int(idx_val)])
    return ''.join(result)
