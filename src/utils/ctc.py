import torch


def decode(prob_matrix, log=False):
    # if log:
    #     return torch.argmin(prob_matrix, dim=1)
    return torch.argmax(prob_matrix, dim=1)

def reduce(sequence: torch.Tensor, blank=0):
    last = sequence[0]
    reduced = [] 
    for i in sequence[1:]:
        if i == last:
            continue
        elif i == blank:
            last = blank
            continue
        else:
            reduced.append(chr(i + ord('A') - 1))
            last = i
    return reduced

def encode(sequence: str):
    return torch.tensor([ord(i) - ord('A') + 1 for i in sequence], dtype=torch.int32)
