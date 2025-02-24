import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ReMinGPT.utils import CfgNode as CN


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    

class CausualSelfActivation(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end. If is possible to use torch.nn.MultiheadAttention
    here but I am including an explicit implementation.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        #Output project
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        #regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # casual mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
