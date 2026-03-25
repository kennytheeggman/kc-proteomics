import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from src.data.data import Peptide
from src.network.network import Model, decode, encode
from src.utils.config import Config

def eval(config: Config, model: Model, train_dataloader: DataLoader[Peptide], eval_dataloader: DataLoader[Peptide]):
    model.eval()
    writer = SummaryWriter(log_dir="runs/eval1")
    global_step = 0
    total_peptides = 0
    correct_peptides = 0
    total_aas = 0
    correct_aas = 0

    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            mass, mz, i, peptide = batch
            batch_sz = config.hyper.batch_size

            spec, prec = model.encode_spectrum.forward((mz.to(config.device), i.to(config.device)))
            embedding = model.encode_embedding.forward(spec, prec)

            # init shape [batch size, max seq len]
            seqs = torch.full((batch_sz, config.seq.max_seq_len), config.PAD_TOKEN)
            seqs[:, 0] = config.SOS_TOKEN

            finished_indices = torch.zeros(batch_sz)

            for i in range(config.seq.max_seq_len):
                pass
                # FLEKSJLJKSLAJfsajjojqf
                # repeat thing and feed back into itself?

            
            # i am so confused oh my goodness gracious
    
    p_correct_seqs = correct_peptides/total_peptides
    p_correct_aas = correct_aas/total_aas

    return p_correct_seqs, p_correct_aas