import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from src.data.data import Peptide
from src.network.network import Model, decode, encode
from src.utils.config import Config

def eval(config: Config, model: Model, train_dataloader: DataLoader[Peptide], eval_dataloader: DataLoader[Peptide], global_step):
    model.eval()
    writer = SummaryWriter(log_dir="runs/eval1")
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
            seqs = torch.full((batch_sz, config.hyper.max_length), config.PAD_TOKEN)
            seqs[:, 0] = config.SOS_TOKEN

            incomplete = torch.full((batch_sz,), True, dtype=torch.bool)  # true for not done, false for done

            for i in range(config.hyper.max_length):
                
                masks = torch.tensor([True] * (i+1) + [False] * (config.hyper.max_length - (i+1)))
                masks = masks.unsqueeze(0).repeat(batch_sz, 1)[incomplete, :]

                peptide_seq = model.encode_sequence.forward((seqs[incomplete, :].to(model.config.device), masks.to(model.config.device)))
                logits = model.decode_sequence.forward((peptide_seq, masks.to(model.config.device)), embedding)

                seqs[incomplete, (i+1)] = 67  # NBMVM THIS IS WRONG wtf how do i even do this

                # put the highest value logit into the next seqs
                # check if any seqs are complete

                # i am confused
                # false true flipped?
                # why is ignore token commented out :(
    
    p_correct_seqs = (correct_peptides/total_peptides) * 100
    p_correct_aas = (correct_aas/total_aas) * 100

    writer.add_scalar("Correct Sequences (%)", p_correct_seqs, global_step)
    writer.add_scalar("Correct AAs (%)", p_correct_aas, global_step)

    model.train()

    return p_correct_seqs, p_correct_aas