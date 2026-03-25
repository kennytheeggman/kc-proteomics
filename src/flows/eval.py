import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from src.data.data import Peptide
from src.network.network import Model, decode, encode
from src.utils.config import Config

def eval(config: Config, model: Model, train_dataloader: DataLoader[Peptide], eval_dataloader: DataLoader[Peptide], global_step):
    model.eval()
    writer = SummaryWriter(log_dir="runs/eval1")
    total_sequences = 0
    correct_sequences = 0
    total_tokens = 0
    correct_tokens = 0

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

                if not incomplete.any():
                    break
                
                masks = torch.tensor([False] * (i+1) + [True] * (config.hyper.max_length - (i+1)))
                masks = masks.unsqueeze(0).repeat(batch_sz, 1)[incomplete, :]

                peptide_seq = model.encode_sequence.forward((seqs[incomplete, :].to(model.config.device), masks.to(model.config.device)))
                logits = model.decode_sequence.forward((peptide_seq, masks.to(model.config.device)), embedding)

                seqs[incomplete, (i+1)] = torch.argmax()  # just need the logit

                # put the highest value logit into the next seqs
                # check if any seqs are complete

                # why is ignore token commented out :( -- for training :(
    
            tgt_seqs = encode_seqs(peptide)
            result_matrix = seqs - tgt_seqs

            # all the ones that r 0 r correct? then do row by row check

            total_tokens = result_matrix.shape[0] * result_matrix.shape[1]
            total_sequences = result_matrix.shape[0]
            
            p_correct_seqs = (correct_sequences/total_sequences) * 100
            p_correct_tokens = (correct_tokens/total_tokens) * 100

            writer.add_scalar("Correct Sequences (%)", p_correct_seqs, global_step)
            writer.add_scalar("Correct Tokens (%)", p_correct_tokens, global_step)

    model.train()

    return p_correct_seqs, p_correct_tokens


def encode_seqs(sequence: list[str], config: Config):
    ENCODE_MAP = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'm': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    result: torch.Tensor = torch.empty(0, config.hyper.max_length)
    for seq in sequence:
        encoded = torch.tensor([config.SOS_TOKEN] + [ENCODE_MAP[aa] for aa in seq] + [config.EOS_TOKEN])
        if (len(encoded) > config.hyper.max_length):
            encoded = encoded[:config.hyper.max_length]
            encoded[-1] = config.EOS_TOKEN
        elif (len(encoded) < config.hyper.max_length):
            encoded = nn.functional.pad(encoded, (0, config.hyper.max_length - len(encoded)), value=config.PAD_TOKEN)
        result = torch.cat([result, encoded.unsqueeze(0)], dim=0)
    return result