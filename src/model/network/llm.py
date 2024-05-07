import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import HoM


class HLMLayer(nn.Module):
    def __init__(self,
                 dim,
                 context_length,
                 order,
                 order_expand,
                 ffw_expand,
                 checkpoint_hom=False):
        super().__init__()
        self.dim = dim
        self.context_length = context_length
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.checkpoint_hom = checkpoint_hom

        self.hom = HoM(dim=dim, order=order, order_expand=order_expand, bias=True)
        self.ffw = nn.Sequential(nn.Linear(dim, ffw_expand * dim, bias=True),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand * dim, dim, bias=True))

    def forward(self, x, mask):
        if self.checkpoint_hom:
            x = x + checkpoint(self.hom, x, x, mask, use_reentrant=False)
        else:
            x = x + self.hom(x, x, mask)
        x = x + self.ffw(x)
        return x

class HLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 dim,
                 n_layers,
                 context_length,
                 order=2,
                 order_expand=2,
                 ffw_expand=2,
                 n_registers=16,
                 checkpoint_hom=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.context_length = context_length
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.n_registers = n_registers
        self.checkpoint_hom = checkpoint_hom

        # self.registers = nn.Parameter(torch.zeros(1, n_registers, dim), requires_grad=True)
        self.layers = nn.ModuleList([HLMLayer(dim, context_length, order, order_expand, ffw_expand, checkpoint_hom=checkpoint_hom) for i in range(n_layers)])
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=True)

        def init_weights_(m):
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights_)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, -np.log(self.context_length))

    def forward(self, x, mask, pos_offset=None): # x has size b x seq_length with b always = 1
        # embed ids
        x_hidden = self.token_emb(x)
        b, n, d = x_hidden.shape

        # add pos embedding
        pos = torch.arange(0, self.context_length).unsqueeze(0)
        if pos_offset is not None:
            pos = pos + pos_offset.unsqueeze(1).to(pos.device) # b x n
        pos = get_1d_sincos_pos_embed_from_grid(self.dim, pos)
        x_hidden = x_hidden + pos.to(x_hidden.device)

        #big loop
        for l in self.layers:
            x_hidden = l(x_hidden, mask)

        # predict
        out = self.output_proj(x_hidden)

        return out

def HLM150M():
    return HLM(vocab_size=32128,
               dim=768,
               n_layers=11,
               context_length=1024,
               order=2,
               order_expand=2,
               ffw_expand=2)

def HLM300M():
    return HLM(vocab_size=32128,
               dim=1024,
               n_layers=14,
               context_length=1024,
               order=2,
               order_expand=2,
               ffw_expand=2)

def HLM500M():
    return HLM(vocab_size=32128,
               dim=1280,
               n_layers=16,
               context_length=2048,
               order=2,
               order_expand=2,
               ffw_expand=2)

def HLM2B():
    return HLM(vocab_size=32128,
               dim=2304,
               n_layers=22,
               context_length=2048,
               order=2,
               order_expand=2,
               ffw_expand=2)

def HLM3B():
    return HLM(vocab_size=32128,
               dim=2560,
               n_layers=28,
               context_length=2048,
               order=2,
               order_expand=2,
               ffw_expand=2)

HLMModels = {
    'HLM150M': HLM150M,
    'HLM500M': HLM500M,
    'HLM2B': HLM2B,
    'HLM3B': HLM3B
}


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (B, M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2).float()
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = torch.einsum('bm,d->bmd', pos, omega)  # (B, M, D/2), outer product

    emb_sin = torch.sin(out) # (B, M, D/2)
    emb_cos = torch.cos(out) # (B, M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], dim=2)  # (B, M, D)
    return emb