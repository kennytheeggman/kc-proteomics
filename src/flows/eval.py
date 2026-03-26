import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from src.data.data import Peptide
from src.network.network import Model, decode, encode
from src.utils.config import Config

def eval(config: Config, model: Model, train_dataloader: DataLoader[Peptide], eval_dataloader: DataLoader[Peptide], global_step, idx, batch):
    model.eval()
    writer = SummaryWriter(log_dir="runs/eval1")
    total_sequences = 0
    correct_sequences = 0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        # for idx, batch in enumerate(eval_dataloader):
            mass, mz, intensity, peptide = batch
            batch_sz = len(peptide)

            spec, prec = model.encode_spectrum.forward((mz.to(config.device), intensity.to(config.device)))
            embedding = model.encode_embedding.forward(spec, prec)

            # init shape [batch size, max seq len]
            seqs = torch.full((batch_sz, config.hyper.max_length), config.PAD_TOKEN)
            seqs[:, 0] = config.SOS_TOKEN

            incomplete = torch.full((batch_sz,), True, dtype=torch.bool)  # true for not done, false for done

            for i in range(config.hyper.max_length-1):

                if not incomplete.any():
                    break
                
                masks = torch.tensor([False] * (i+1) + [True] * (config.hyper.max_length - (i+1)))
                masks = masks.unsqueeze(0).repeat(batch_sz, 1)[incomplete, :]

                peptide_seq = model.encode_sequence.forward((seqs[incomplete, :].float().to(model.config.device), masks.to(model.config.device)))
                logits = model.decode_sequence.forward((peptide_seq, masks.to(model.config.device)), embedding[incomplete])

                next = torch.argmax(logits[:, i, :], dim=-1).cpu()
                seqs[incomplete, (i+1)] = next  # just need the logit

                complete_i = (next != config.EOS_TOKEN)
                incomplete[incomplete.clone()] = complete_i

                # look at ignore training token !! reminder (unrelated to this file)

            tgt_seqs = encode_seqs(peptide, config)
            scores = seqs - tgt_seqs

            non_pad = ~((seqs == config.PAD_TOKEN) & (tgt_seqs == config.PAD_TOKEN))  # if we want to not count pad tokens

            results_tokens = (scores == 0) & non_pad
            score_per_seq = torch.sum(results_tokens, dim=1)
            correct_tokens += torch.sum(score_per_seq)

            results_sequences = (score_per_seq == config.hyper.max_length)
            correct_sequences += torch.sum(results_sequences)

            total_tokens += torch.sum(non_pad)
            total_sequences += scores.shape[0]
            
            p_correct_seqs = (correct_sequences/total_sequences) * 100
            p_correct_tokens = (correct_tokens/total_tokens) * 100

            writer.add_scalar("Correct Sequences (%)", p_correct_seqs, global_step)
            writer.add_scalar("Correct Tokens (%)", p_correct_tokens, global_step)

    model.train()

    return p_correct_seqs, p_correct_tokens, logits


def encode_seqs(sequence: list[str], config: Config):
    ENCODE_MAP = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'm': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    result: torch.Tensor = torch.empty(0, config.hyper.max_length, dtype=torch.long)
    for seq in sequence:
        encoded = torch.tensor([config.SOS_TOKEN] + [ENCODE_MAP[aa] for aa in seq] + [config.EOS_TOKEN])
        if (len(encoded) > config.hyper.max_length):
            encoded = encoded[:config.hyper.max_length]
            encoded[-1] = config.EOS_TOKEN
        elif (len(encoded) < config.hyper.max_length):
            encoded = nn.functional.pad(encoded, (0, config.hyper.max_length - len(encoded)), value=config.PAD_TOKEN)
        result = torch.cat([result, encoded.unsqueeze(0)], dim=0)
    return result