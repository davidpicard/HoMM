import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import HoMLayer
from timm.models.vision_transformer import Attention


def modulation(x, scale, bias):
    return x * (1+scale) + bias


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mha = Attention(dim, num_heads=n_heads, qkv_bias=True)
        self.ffw_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffw = nn.Sequential(nn.Linear(dim, 4 * dim, bias=True),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim, bias=True))
        self.cond_mlp = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 4 * dim, bias=True))
        self.gate_mlp = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 2 * dim, bias=True))


    def forward(self, x, c):
        s1, b1, s2, b2 = self.cond_mlp(c).chunk(4, -1)
        g1, g2 = self.gate_mlp(c).chunk(2, -1)

        # mha
        x_ln = modulation(self.mha_ln(x), s1.unsqueeze(1), b1.unsqueeze(1))
        x = x + self.mha(x_ln)*(1+g1.unsqueeze(1))

        #ffw
        x_ln = modulation(self.ffw_ln(x), s2.unsqueeze(1), b2.unsqueeze(1))
        x = x + self.ffw(x_ln)*(1+g2.unsqueeze(1))

        return x

from diffusers.models.transformer_2d import Transformer2DModelOutput

class ClassConditionalDiT(nn.Module):
    def __init__(self,
                 n_classes: int,
                 n_timesteps: int,
                 im_size: int,
                 kernel_size: int,
                 dim: int,
                 n_layers,
                 order=2,
                 order_expand=4,
                 ffw_expand=4,
                 dropout=0.):
        super().__init__()
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout

        self.classes_emb = nn.Embedding(n_classes + 1, dim)
        self.freqs = nn.Parameter(torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim//2) / dim), requires_grad=False)
        self.time_emb = nn.Sequential(nn.Linear(dim, 4*dim, bias=True),
                                      nn.SiLU(),
                                      nn.Linear(4*dim, dim, bias=True)
                                      )
        self.n_patches = (im_size // kernel_size)
        # for diffusers
        self.in_channels = 3
        self.sample_size = (self.n_patches, self.n_patches)
        self.pos_emb = nn.Parameter(torch.zeros((1, self.n_patches ** 2, dim)), requires_grad=False)
        self.in_conv = nn.Conv2d(3, dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
        self.layers = nn.ModuleList(
            [DiTBlock(dim, order) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_size * kernel_size * 3, bias=True)
        self.out_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

        # init
        # layers
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights_)
        # zeros modulation gates
        for l in self.layers:
            m = l.cond_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            m = l.gate_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, -1.)
        #pos emb
        pos_emb = get_2d_sincos_pos_embed(self.pos_emb.shape[-1], self.n_patches)
        self.pos_emb.data.copy_(torch.from_numpy(pos_emb).float().unsqueeze(0))
        # patch and time emb
        nn.init.normal_(self.classes_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)



    def forward(self, img, time, cls, return_dict: bool = True,):

        # patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t)
        # cond
        c = self.classes_emb(cls)
        c = c+t

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, c)
        s, b = self.out_mod(c).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s.unsqueeze(1), b.unsqueeze(1))
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(out, 'b (h w) (k s c) -> b c (h k) (w s)',
                               h=self.n_patches, k=self.kernel_size, c=3)

        return out


class ClassConditionalHoMDiffusion(nn.Module):
    def __init__(self,
                 n_classes: int,
                 n_timesteps: int,
                 im_size: int,
                 kernel_size: int,
                 dim: int,
                 n_layers,
                 order=2,
                 order_expand=4,
                 ffw_expand=4,
                 dropout=0.):
        super().__init__()
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout

        self.classes_emb = nn.Parameter(torch.zeros((n_layers, n_classes, dim)), requires_grad=True)
        self.time_emb = nn.Parameter(torch.zeros((n_layers, n_timesteps+1, dim)), requires_grad=True)

        self.n_patches = (im_size//kernel_size)
        self.pos_emb = nn.Parameter(torch.zeros((1, self.n_patches**2, dim)))
        # self.n_registers = 16
        # self.register = nn.Parameter(torch.zeros(1, self.n_registers, dim), requires_grad=True)

        self.in_conv = nn.Conv2d(3, dim, kernel_size=kernel_size, stride=kernel_size)
        # self.sa_layers = nn.ModuleList([HoMLayer(dim, order, order_expand, ffw_expand, dropout) for i in range(n_layers)])
        # self.ca_layers = nn.ModuleList(
        #     [HoMLayer(dim, order, order_expand, ffw_expand, dropout) for i in range(n_layers)])
        self.sa_layers = nn.ModuleList(
            [DiTBlock(dim, n_classes, order, n_timesteps) for i in range(n_layers)])
        # self.ca_layers = nn.ModuleList(
        #     [AttentionModule(dim, order, order_expand, ffw_expand, dropout) for i in range(n_layers)])
        self.out_ln = nn.LayerNorm(dim)
        self.outproj = nn.Linear(dim, kernel_size*kernel_size*3)

        # init
        nn.init.trunc_normal_(
            self.classes_emb, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
        nn.init.trunc_normal_(
            self.time_emb, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
        nn.init.trunc_normal_(
            self.pos_emb, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
        # nn.init.trunc_normal_(
        #     self.register, std=0.02, a=-2 * 0.02, b=2 * 0.02
        # )
        self.apply(self.init_weights_)

    def init_weights_(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, img, cls, time):
        # quantize time
        time = time * self.n_timesteps
        tq = torch.clamp(time.floor().long(), 0, self.n_timesteps)
        t2 = (time - tq).unsqueeze(1)
        t1 = 1 - t2

        #patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # registers
        # r = self.register * torch.ones((b, 1, 1)).to(x.device)

        # forward pass
        for l in range(self.n_layers):
            #CA
            c = self.classes_emb[l, cls.argmax(dim=1)]
            t = (t1*self.time_emb[l, tq, ...] + t2*self.time_emb[l, tq+1, ...])
            # ctx = torch.cat([t, c], dim=1)
            # r = self.ca_layers[l](r, xc=ctx) # registers read from ctx
            # sa
            # x = torch.cat([r, x], dim=1)
            # x = self.sa_layers[l](x) # both r and x read from everything
            # x = x[:,self.n_registers:, :]
            # r = x[:, :self.n_registers, :]

            # dit
            x = self.sa_layers[l](x, c, t)

        # depatchify
        out = self.outproj(self.out_ln(x))
        out = einops.rearrange(out, 'b (h w) (k s c) -> b c (h k) (w s)',
                               h=self.n_patches, k=self.kernel_size, c=3)

        return (out,)


# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb