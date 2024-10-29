import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import HoM


def modulation(x, scale, bias):
    return x * (1+scale) + bias



class MMDiHBlock(nn.Module):
    def __init__(self, dim: int, order: int, order_expand: int, ffw_expand: int):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.mha_ln_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mha_ln_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.pre_lin_x = nn.Linear(dim, dim, bias=True)
        self.pre_lin_c = nn.Linear(dim, dim, bias=True)
        self.hom = HoM(dim, order=order, order_expand=order_expand, bias=True)
        self.ffw_ln_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffw_ln_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.post_lin_x = nn.Linear(dim, dim, bias=True)
        self.post_lin_c = nn.Linear(dim, dim, bias=True)
        self.ffw_x = nn.Sequential(nn.Linear(dim, ffw_expand * dim, bias=True),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand * dim, dim, bias=True))
        self.ffw_c = nn.Sequential(nn.Linear(dim, ffw_expand * dim, bias=True),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand * dim, dim, bias=True))
        self.cond_mlp_x = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 4 * dim, bias=True))
        self.gate_ml_x = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 2 * dim, bias=True))
        self.cond_mlp_c = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 4 * dim, bias=True))
        self.gate_ml_c = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 2 * dim, bias=True))


    def forward(self, x, t, c):
        xs1, xb1, xs2, xb2 = self.cond_mlp_x(t).chunk(4, -1)
        xg1, xg2 = self.gate_mlp_x(t).chunk(2, -1)
        cs1, cb1, cs2, cb2 = self.cond_mlp_c(t).chunk(4, -1)
        cg1, cg2 = self.gate_mlp_c(t).chunk(2, -1)

        # mha
        x_ln = modulation(self.mha_ln(x), xs1, xb1)
        b, n, d = x_ln.shape
        c_ln = modulation(self.mha_ln_c(c), cs1, cb1)
        h = self.hom(torch.cat([self.pre_lin_x(x_ln), self.pre_lin_c(c_ln)], dim=1))
        hx = self.post_lin_x(h[:,0:n, :])
        hc = self.post_lin_c(h[:,n:-1, :])
        x = x + hx * (1 + xg1)
        c = c + hc * (1 + cg1)

        #ffw
        x_ln = self.ffw_ln_x(modulation(x, xs2, xb2))
        c_ln = self.ffw_ln_c(modulation(c, cs2, cb2))
        x = x + self.ffw_x(x_ln)*(1+xg2)
        c = c + self.ffw_c(c_ln)*(1+cg2)

        return x, c

class TextConditionalMMDiH(nn.Module):
    def __init__(self,
                 input_dim: int,
                 text_dim: int,
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
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout

        self.text_emb = nn.Linear(text_dim, dim)
        self.freqs = nn.Parameter(torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim//2) / dim), requires_grad=False)
        self.time_emb = nn.Sequential(nn.Linear(dim, 4*dim, bias=True),
                                      nn.SiLU(),
                                      nn.Linear(4*dim, dim, bias=True)
                                      )
        self.n_patches = (im_size // kernel_size)
        # for diffusers
        self.in_channels = input_dim
        self.sample_size = (self.n_patches, self.n_patches)
        self.in_conv = nn.Conv2d(input_dim, dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
        self.layers = nn.ModuleList(
            [MMDiHBlock(dim=dim, order=order, order_expand=order_expand, ffw_expand=ffw_expand) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_size * kernel_size * input_dim, bias=True)
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
        # patch and time emb
        nn.init.normal_(self.classes_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)

    def forward(self, img, time, cls):

        # patchify
        x = self.in_conv(img)
        b, c, h, w = x.shape
        pos_emb = sincos_embedding_3d(self.n_patches, self.n_patches, self.dim).to(x.device)
        pos_emb = einops.rearrange(pos_emb, "b h w d -> b (h w) d")

        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        b, n, d = x.shape
        x = x + pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t)
        # cond
        c = self.classes_emb(cls)
        c = c+t
        c = c.unsqueeze(1)

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, c)
        s, b = self.out_mod(c).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s, b)
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(out, 'b (h w) (k s c) -> b c (h k) (w s)',
                               h=self.n_patches, k=self.kernel_size, s=self.kernel_size)

        return out


## -------------------------------------------------------------

class TextVideoDiHBlock(nn.Module):
    def __init__(self, dim: int, order: int, order_expand: int, ffw_expand: int):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.x_mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.c_mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.hom = HoM(dim, order=order, order_expand=order_expand, bias=True)
        self.c_hom = HoM(dim, order=order, order_expand=order_expand, bias=True)
        self.ffw_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffw = nn.Sequential(nn.Linear(dim, ffw_expand * dim, bias=True),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand * dim, dim, bias=True))
        self.cond_mlp = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 8 * dim, bias=True))
        self.gate_mlp = nn.Sequential(
                                 nn.SiLU(),
                                 nn.Linear(dim, 3 * dim, bias=True))


    def forward(self, x, t, c, mask, temporal_mask=None):
        sx, bx, sc, bc, s1, b1, s2, b2 = self.cond_mlp(t).chunk(8, -1)
        gc, g1, g2 = self.gate_mlp(t).chunk(3, -1)

        # ca
        x_ln = modulation(self.x_mha_ln(x), sx, bx)
        c_ln = modulation(self.c_mha_ln(c), sc, bc)
        x = x + self.c_hom(x_ln, c_ln, mask) * (1 + gc)

        # sa
        x_ln = modulation(self.mha_ln(x), s1, b1)
        x = x + self.hom(x_ln, mask=temporal_mask) * (1 + g1)
        # x = x + checkpoint(self.hom,x_ln, use_reentrant=False)*(1+g1)

        #ffw
        x_ln = modulation(self.ffw_ln(x), s2, b2)
        x = x + self.ffw(x_ln)*(1+g2)

        return x


class TextVideoDiH(nn.Module):
    def __init__(self,
                 input_dim: int,
                 text_dim: int,
                 n_timesteps: int,
                 vid_size: str,
                 vid_length: int,
                 kernel_s: int,
                 kernel_t: int,
                 dim: int,
                 n_layers,
                 order=2,
                 order_expand=4,
                 ffw_expand=4,):
        super().__init__()
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.n_timesteps = n_timesteps
        vid_size = vid_size.split("x")
        vid_size = (int(vid_size[0]), int(vid_size[1]))
        self.vid_size = vid_size
        self.vid_length = vid_length
        self.kernel_s = kernel_s
        self.kernel_t = kernel_t
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.text_emb = nn.Linear(text_dim, dim)
        self.freqs = nn.Parameter(torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim//2) / dim), requires_grad=False)
        self.time_emb = nn.Sequential(nn.Linear(dim, 4*dim, bias=True),
                                      nn.SiLU(),
                                      nn.Linear(4*dim, dim, bias=True)
                                      )
        self.n_patches_h = (vid_size[0] // kernel_s)
        self.n_patches_w = (vid_size[1] // kernel_s)
        self.n_frames =  (vid_length // kernel_t)
        # for diffusers
        self.in_channels = input_dim
        self.sample_size = (self.n_patches_h, self.n_patches_w)
        self.in_proj = nn.Linear(input_dim*kernel_s*kernel_s*kernel_t, dim)
        self.layers = nn.ModuleList(
            [TextVideoDiHBlock(dim=dim, order=order, order_expand=order_expand, ffw_expand=ffw_expand) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_t * kernel_s * kernel_s * input_dim, bias=True)
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
        # patch and time emb
        nn.init.normal_(self.text_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)

    def forward(self, vid, time, txt, mask, temporal_mask=None):
        b, c, t, h, w = vid.shape
        # print(f"vid: {vid.shape}")

        # patchify
        x = einops.rearrange(vid, "b c (t s) (h k) (w l) -> b (t h w) (s k l c)", s=self.kernel_t, k=self.kernel_s, l=self.kernel_s)
        x = self.in_proj(x)
        pos_emb = sincos_embedding_3d(self.n_frames, self.n_patches_h, self.n_patches_w, self.dim).to(x.device)
        pos_emb = einops.rearrange(pos_emb, "b t h w d -> b (t h w) d")
        x = x + pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t).unsqueeze(1)
        # cond
        c = self.text_emb(txt)

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, t, c, mask, temporal_mask=temporal_mask)
        s, b = self.out_mod(t).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s, b)
        # out = x
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(out, 'b (t h w) (s k l c) -> b c (t s) (h k) (w l)',
                               h=self.n_patches_h, w=self.n_patches_w, k=self.kernel_s, l=self.kernel_s, s=self.kernel_t)

        return out

    def make_block_causal_temporal_mask(self):
        total_tokens = self.n_patches_h*self.n_patches_w*self.n_frames
        frame_tokens = self.n_patches_h*self.n_patches_w
        mask = torch.zeros(1, total_tokens, total_tokens)
        for f in range(self.n_frames):
            mask[:, f*frame_tokens:(f+1)*frame_tokens, 0:(f+1)*frame_tokens] = 1
        return mask


import math
def sincos_embedding_3d(t, h, w, d, r=0):
    freqs = torch.linspace(0.5, 2, d//4)
    f = einops.rearrange(freqs, "(b n d) -> b n d", b=1, n=1)

    n = torch.linspace(0, 1, t)
    n = einops.rearrange(n, "(b n d) -> b n d", b=1, d=1)
    s = (n * f * math.pi+f).sin()
    c = (n * f * math.pi+f).cos()
    n = torch.cat([s, c], dim=-1)
    n = einops.rearrange(n, "b (n h w) d -> b n h w d", h=1, w=1).repeat([1, 1, h, w, 1])


    freqs = torch.linspace(0.5, 2, d//8)
    f = einops.rearrange(freqs, "(b n d) -> b n d", b=1, n=1)

    x = torch.linspace(0, 1, w)
    x = einops.rearrange(x, "(b n d) -> b n d", b=1, d=1)
    s = (x * f * math.pi+f).sin()
    c = (x * f * math.pi+f).cos()
    x = torch.cat([s, c], dim=-1)
    x = einops.rearrange(x, "b (n h w) d -> b n h w d", n=1, h=1).repeat([1, t, h, 1, 1])

    y = torch.linspace(0, 1, h)
    y = einops.rearrange(y, "(b n d) -> b n d", b=1, d=1)
    s = (y * f * math.pi+f).sin()
    c = (y * f * math.pi+f).cos()
    y = torch.cat([s, c], dim=-1)
    y = einops.rearrange(y, "b (n h w) d -> b n h w d", n=1, w=1).repeat([1, t, 1, w, 1])

    # print(f"n: {n.shape} x: {x.shape} y: {y.shape}")
    return torch.cat([n, x, y], dim=-1)