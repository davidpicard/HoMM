import einops
import torch
import torch.nn as nn
from .layers import HoMLayer


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
        self.register = nn.Parameter(torch.zeros(1, 4, dim), requires_grad=True)

        self.in_conv = nn.Conv2d(3, dim, kernel_size=kernel_size, stride=kernel_size)
        self.layers = nn.ModuleList([HoMLayer(dim, order, order_expand, ffw_expand, dropout) for i in range(n_layers)])
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
        nn.init.trunc_normal_(
            self.register, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
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
        # add registers
        x = torch.cat([self.register* torch.ones((b, 1, 1)).to(x.device), x], dim=1)

        # forward pass
        for l in range(self.n_layers):
            c = self.classes_emb[l, cls.argmax(dim=1)].unsqueeze(1)
            t = (t1*self.time_emb[l, tq, ...] + t2*self.time_emb[l, tq+1, ...]).unsqueeze(1)
            x = torch.cat([t, c, x], dim=1)
            x = self.layers[l](x)
            x = x[:,2:, :]

        #remove register
        x = x[:, 4:, :]

        # depatchify
        out = self.outproj(x)
        out = einops.rearrange(out, 'b (h w) (k s c) -> b c (h k) (w s)',
                               h=self.n_patches, k=self.kernel_size, c=3)

        return out


