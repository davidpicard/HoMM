import einops
import numpy as np
import torch
import torch.nn as nn
from .layers import HoM


## -------------------------------------------------------------

class TextImageDiHBlock(nn.Module):
    def __init__(self, dim: int, order: int, order_expand: int, ffw_expand: int):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.x_mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.c_mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.hom = HoM(dim, order=order, order_expand=order_expand, bias=False)
        self.c_hom = HoM(dim, order=order, order_expand=order_expand, bias=False)
        self.ffw = nn.Sequential(nn.Linear(dim, ffw_expand * dim, bias=True),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand * dim, dim, bias=True))
        self.gs = nn.Parameter(-0.5*torch.ones(1, 1, dim), requires_grad=True)
        self.gc = nn.Parameter(-0.5*torch.ones(1, 1, dim), requires_grad=True)
        self.gf = nn.Parameter(-0.5*torch.ones(1, 1, dim), requires_grad=True)


    def forward(self, x, c, mask):
        # sa
        x_ln = self.mha_ln(x)
        x = x + self.hom(x_ln) * (1+self.gs)
        # ca
        x_ln = self.x_mha_ln(x)
        c_ln = self.c_mha_ln(c)
        x = x + self.c_hom(x_ln, c_ln, mask) * (1+self.gc)
        #ffw
        x = x + self.ffw(x)*(1+self.gf)
        return x


class TextImageDiH(nn.Module):
    def __init__(self,
                 input_dim: int,
                 text_dim: int,
                 n_timesteps: int,
                 img_size: int,
                 kernel_s: int,
                 dim: int,
                 n_layers,
                 order=2,
                 order_expand=4,
                 ffw_expand=4,
                 register=64):
        super().__init__()
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.n_timesteps = n_timesteps
        img_size = (img_size//8, img_size//8)
        self.kernel_s = kernel_s
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.n_register = register
        self.register = nn.Parameter(torch.randn(1, self.n_register, dim), requires_grad=True)
        self.text_emb = nn.Sequential(nn.Linear(text_dim, 4*dim, bias=True),
                                      nn.GELU(),
                                      nn.Linear(4*dim, dim, bias=True)
                                      )
        self.freqs = nn.Parameter(torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim//2) / dim), requires_grad=False)
        self.time_emb = nn.Sequential(nn.Linear(dim, 4*dim, bias=True),
                                      nn.GELU(),
                                      nn.Linear(4*dim, dim, bias=True)
                                      )
        # self.pos_mlp = nn.Sequential(nn.Linear(dim, 4*dim, bias=True),
        #                               nn.GELU(),
        #                               nn.Linear(4*dim, dim, bias=True)
        #                               )
        self.n_patches_h = (img_size[0] // kernel_s)
        self.n_patches_w = (img_size[1] // kernel_s)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches_h, self.n_patches_w, dim), requires_grad=True)
        # for diffusers
        self.in_channels = input_dim
        self.sample_size = (self.n_patches_h, self.n_patches_w)
        self.in_proj = nn.Linear(input_dim*kernel_s*kernel_s, dim)
        self.layers = nn.ModuleList(
            [TextImageDiHBlock(dim=dim, order=order, order_expand=order_expand, ffw_expand=ffw_expand) for _ in range(n_layers)])
        self.out_proj = nn.Linear(dim, kernel_s * kernel_s * input_dim, bias=True)

        # init
        # layers
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                # fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                # nn.init.normal_(m.weight, std=0.5/np.sqrt(fan_in + fan_out))
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights_)
        # patch, text and time emb
        # nn.init.normal_(self.pos_mlp[0].weight, std=0.02)
        # nn.init.normal_(self.pos_mlp[2].weight, std=0.02)
        nn.init.trunc_normal_(self.pos, 0.0, 0.02)
        nn.init.normal_(self.text_emb[0].weight, std=0.02)
        nn.init.normal_(self.text_emb[2].weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, img, time, txt, mask, temporal_mask=None):
        b, c, h, w = img.shape

        # patchify
        x = einops.rearrange(img, "b c (h k) (w l) -> b (h w) (k l c)", k=self.kernel_s, l=self.kernel_s)
        x = self.in_proj(x)
        # pos_emb = sincos_embedding_2d(self.n_patches_h, self.n_patches_w, self.dim).to(x.device)
        # pos_emb = einops.rearrange(pos_emb, "b h w d -> b (h w) d")
        # pos_emb = self.pos_mlp(pos_emb)
        pos_emb = einops.rearrange(self.pos, "b h w d -> b (h w) d")
        x = x + pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # registers
        r = self.register * torch.ones(b, 1, 1).to(img.device)
        x = torch.cat([r, x], dim=1)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t).unsqueeze(1)
        # cond
        c = self.text_emb(txt)
        # add time embedding
        c = torch.cat([t, c],dim=1)
        mask = torch.cat([torch.ones(b, 1).to(mask.device), mask], dim=1)

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, c, mask)
        out = x
        out = self.out_proj(out)[:, self.n_register:, :]

        # depatchify
        out = einops.rearrange(out, 'b (h w) (k l c) -> b c (h k) (w l)',
                               h=self.n_patches_h, w=self.n_patches_w, k=self.kernel_s, l=self.kernel_s)

        return out

    def make_block_causal_temporal_mask(self):
        total_tokens = self.n_patches_h*self.n_patches_w*self.n_frames
        frame_tokens = self.n_patches_h*self.n_patches_w
        mask = torch.zeros(1, total_tokens, total_tokens)
        for f in range(self.n_frames):
            mask[:, f*frame_tokens:(f+1)*frame_tokens, 0:(f+1)*frame_tokens] = 1
        return mask

def TIDiH_S2(**kwargs):
    return TextImageDiH(input_dim=4,
                        text_dim=2048,
                        n_timesteps=1000,
                        kernel_s=2,
                        dim=384,
                        n_layers=12,
                        order=2,
                        order_expand=2,
                        ffw_expand=2,
                        **kwargs)

def TIDiH_B2(**kwargs):
    return TextImageDiH(input_dim=4,
                        text_dim=2048,
                        n_timesteps=1000,
                        kernel_s=2,
                        dim=768,
                        n_layers=12,
                        order=2,
                        order_expand=2,
                        ffw_expand=2,
                        **kwargs)

def TIDiH_M2(**kwargs):
    return TextImageDiH(input_dim=4,
                        text_dim=2048,
                        n_timesteps=1000,
                        kernel_s=2,
                        dim=768,
                        n_layers=16,
                        order=2,
                        order_expand=2,
                        ffw_expand=2,
                        **kwargs)

def TIDiH_L2(**kwargs):
    return TextImageDiH(input_dim=4,
                        text_dim=2048,
                        n_timesteps=1000,
                        kernel_s=2,
                        dim=1024,
                        n_layers=16,
                        order=2,
                        order_expand=2,
                        ffw_expand=2,
                        **kwargs)


def TIDiH_XL2(**kwargs):
    return TextImageDiH(input_dim=4,
                        text_dim=2048,
                        n_timesteps=1000,
                        kernel_s=2,
                        dim=1024,
                        n_layers=20,
                        order=2,
                        order_expand=2,
                        ffw_expand=2,
                        **kwargs)

TVDiH_models = {
    'TVDiH_S2': TIDiH_S2,
    'TVDiH_B2': TIDiH_B2,
    'TVDiH_M2': TIDiH_M2,
    'TVDiH_L2': TIDiH_L2,
    'TVDiH_XL2': TIDiH_XL2,
}

import math
def sincos_embedding_2d(h, w, d, r=0):

    freqs = torch.linspace(0.5, 2, d//4)
    f = einops.rearrange(freqs, "(b n d) -> b n d", b=1, n=1) # 1 x 1 x d//4

    x = torch.linspace(0, 1, w)
    x = einops.rearrange(x, "(b n d) -> b n d", b=1, d=1) # 1 x w x 1
    s = (x * f * math.pi+f).sin() # 1 x w x d//4
    c = (x * f * math.pi+f).cos() # 1 x w x d//4
    x = torch.cat([s, c], dim=-1) # 1 x w x d//2
    x = einops.rearrange(x, "b (h w) d -> b h w d", h=1).repeat([1, h, 1, 1])

    y = torch.linspace(0, 1, h)
    y = einops.rearrange(y, "(b n d) -> b n d", b=1, d=1)
    s = (y * f * math.pi+f).sin()
    c = (y * f * math.pi+f).cos()
    y = torch.cat([s, c], dim=-1)
    y = einops.rearrange(y, "b (h w) d -> b h w d", w=1).repeat([1, 1, w, 1])

    # print(f"n: {n.shape} x: {x.shape} y: {y.shape}")
    return torch.cat([x, y], dim=-1)