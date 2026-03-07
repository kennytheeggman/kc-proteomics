import torch
import math


def decode(probabilities, masses, precursor_m, tol, mass_n):

    assert isinstance(mass_n, int)

    max_seq_len = probabilities.shape[0]

    dm = precursor_m/mass_n
    padding_cols = math.floor(tol/dm)
    n_cols = mass_n + padding_cols

    dp = torch.full((max_seq_len, n_cols), float('-inf'))
    parent = torch.zeros((max_seq_len, n_cols, 2))
    dp[0, 0] = 0

    # fill out dynamic programming score array
    for i in range(max_seq_len-1):
        for m in range(n_cols):
            if dp[i][m] > float('-inf'):
                for a in range(len(masses)):
                    
                    aa = masses[a]

                    cumulative_mass = m*dm + aa
                    if cumulative_mass > dm*n_cols:
                        continue
                    discretized_mass = torch.round(cumulative_mass/dm).long()
                    discretized_mass = torch.clamp(discretized_mass, 0, n_cols-1)

                    score = dp[i, m] + probabilities[i, a]  # prob[i], because dp is one idx ahead
                    if score > dp[i+1, discretized_mass]:
                        dp[i+1, discretized_mass] = score
                        parent[i+1, discretized_mass] = torch.tensor([m, a]) # store prev mass and next aa index

    # slice out target mass scores, within tol, and find best
    best_score = float('-inf')
    pos = None  # can be any of the discrete masses within tolerance of precursor mass
    lowest = max(0, mass_n - padding_cols - 1)
    for i in range(lowest, n_cols):
        for j in range(max_seq_len):
            if dp[j, i] > best_score:
                best_score = dp[j, i]
                pos = (j, i)
                        
    # retrieve aa sequence from parent
    seq = torch.empty((pos[0] + 1))  # maybe best to pre-allocate memory since we know the size
    if pos is None:
        print("did not find")
    else:
        cur_mass = pos[1]
        for i in range(pos[0], 0, -1):
            m, a = parent[i, cur_mass]  # m is idx
            m = int(m.item())
            seq[i] = a
            cur_mass -= m
    return seq[1:-1]

def reduce(sequence: torch.Tensor, blank=0):
    last = sequence[0]
    reduced = [last] 
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
