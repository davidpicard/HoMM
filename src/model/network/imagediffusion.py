import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import HoMLayer


def modulation(x, scale, bias):
    return x * (1+scale.unsqueze(1)) + bias.unsqueeze(1)

class AttentionModule(nn.Module):
    def __init__(self, dim, order, order_expand, ffw_expand, dropout=0.):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.q = nn.Linear(dim, dim*order_expand)
        self.k = nn.Linear(dim, dim*order_expand)
        self.v = nn.Linear(dim, dim*order_expand)
        self.out_proj = nn.Linear(dim*order_expand, dim)

        self.mha = nn.MultiheadAttention(embed_dim=dim*order_expand, num_heads=order, batch_first=True)
        self.ffw = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(dim, ffw_expand * dim),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand * dim, dim))

    def forward(self, x, xc=None, mask=None):
        if xc is None:
            xc = x

        q = self.q(x)
        k = self.k(xc)
        v = self.v(xc)
        mha = self.mha(q, k, v)[0]
        x = x + self.out_proj(mha)

        # ffw
        x = x + self.ffw(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, n_classes, n_heads, n_timesteps=250):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.n_timesteps = n_timesteps

        self.mha_ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads=n_heads)
        self.ffw_ln = nn.LayerNorm(dim)
        self.ffw = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, 4 * dim),
                                 nn.GELU(),
                                 nn.LayerNorm(4 * dim),
                                 nn.Linear(4 * dim, dim))
        self.cond_mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, 4 * dim),
                                 nn.GELU(),
                                 nn.LayerNorm(4 * dim),
                                 nn.Linear(4 * dim, 6))

    def init_modulation_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, c, t):
        g1, s1, b1, g2, s2, b2 = self.cond_mlp(c+t).chunk(6, -1)

        # mha
        x_ln = modulation(self.mha_ln(x), s1, b1)
        x = x + self.mha(x_ln, x_ln, x_ln)[0]*g1

        #ffw
        x_ln = modulation(self.ffw_ln(x), s2, b2)
        x = x + self.ffw(x_ln)*g2

        return x

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

        self.classes_emb = nn.Parameter(torch.zeros((n_layers, n_classes, dim)), requires_grad=True)
        self.time_emb = nn.Parameter(torch.zeros((n_layers, n_timesteps + 1, dim)), requires_grad=True)

        self.n_patches = (im_size // kernel_size)
        self.pos_emb = nn.Parameter(torch.zeros((1, self.n_patches ** 2, dim)))

        self.in_conv = nn.Conv2d(3, dim, kernel_size=kernel_size, stride=kernel_size)
        self.layers = nn.ModuleList(
            [DiTBlock(dim, n_classes, order, n_timesteps) for i in range(n_layers)])
        self.out_ln = nn.LayerNorm(dim)
        self.outproj = nn.Linear(dim, kernel_size * kernel_size * 3)

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
        self.apply(self.init_weights_)
        for l in self.layers:
            l.apply(l.init_modulation_)

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

        # patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # forward pass
        for l in range(self.n_layers):
            # CA
            c = self.classes_emb[l, cls.argmax(dim=1)]
            t = (t1 * self.time_emb[l, tq, ...] + t2 * self.time_emb[l, tq + 1, ...])
            x = self.layers[l](x, c, t)

        # depatchify
        out = self.outproj(self.out_ln(x))
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

        return out


