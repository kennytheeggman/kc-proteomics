import torch
from torch import nn

class FeedForward(nn.Module):  # no layer normalization, but should be fine

    # hidden_dims is [size of hidden layer 1, size of hidden layer 2, etc]
    def __init__(self, input_dim, output_dim, hidden_dims, normalizer=nn.ReLU, bias=True, dropout=0):
        super().__init__()
        linear_stack = nn.ModuleList([])
        if len(hidden_dims) == 0:
            linear_stack.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            linear_stack.append(nn.Linear(input_dim, hidden_dims[0], bias=bias))
            if normalizer is not None:
                linear_stack.append(normalizer())
            linear_stack.append(nn.Dropout(p=dropout))
            for i in range(len(hidden_dims)-1):
                linear_stack.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=bias))
                if normalizer is not None:
                    linear_stack.append(normalizer())
                linear_stack.append(nn.Dropout(p=dropout))
            linear_stack.append(nn.Linear(hidden_dims[-1], output_dim, bias=bias))
        self.linear_stack = nn.Sequential(*linear_stack)
    
    def forward(self, x):
        return self.linear_stack(x)