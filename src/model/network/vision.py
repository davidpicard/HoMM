import einops
import torch
import torch.nn as nn
from .layers import HoMLayer

class HoMVision(nn.Module):
    def __init__(self, nb_classes, dim=256, im_size=256, kernel_size=16, nb_layers=12, order=4, order_expand=8,
                 ffw_expand=4, dropout=0., pooling='cls', in_conv=True):
        super().__init__()
        self.nb_classes = nb_classes
        self.dim = dim
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.nb_layers = nb_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.pooling = pooling

        self.conv = None
        if in_conv:
            self.conv = nn.Conv2d(3, dim, kernel_size=kernel_size, stride=kernel_size)
        self.layers = nn.ModuleList([HoMLayer(dim, order, order_expand, ffw_expand, dropout=dropout) for i in range(nb_layers)])
        self.out_proj = nn.Linear(dim, nb_classes)

        n = (im_size//kernel_size)**2
        self.position = nn.Parameter(torch.randn((1, n, dim)), requires_grad=True)
        nn.init.trunc_normal_(
            self.position, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
        self.cls = nn.Parameter(torch.randn((1, 1, dim)), requires_grad=True)
        nn.init.trunc_normal_(
            self.cls, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )

        # init
        self.apply(self.init_weights_)
        nn.init.zeros_(self.out_proj.weight)
        if pooling == 'cls':
            nn.init.constant_(self.out_proj.bias, -6.9)
        else:
            nn.init.zeros_(self.out_proj.bias)


    def init_weights_(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):

        if self.conv is not None:
            x = self.conv(x)
            x = einops.rearrange(x, 'b c h w -> b (h w) c')

        b, n, c = x.shape
        ones = torch.ones(b, 1, 1).to(x.device)
        if n == self.position.shape[1]:
            position = self.position
        else:
            position = einops.rearrange(self.position, "b (m n) d ->b d m n", m=self.im_size//self.kernel_size)
            m = torch.sqrt(torch.tensor(n)).int()
            position = nn.functional.interpolate(position, size=(m, m), mode='bicubic')
            position = einops.rearrange(position, "b d m n -> b (m n) d")
        x = x + position * ones
        cls = self.cls * ones

        x = torch.cat([cls, x], dim=1)
        if mask is not None:
            mask = torch.cat([torch.ones(b, 1).to(mask.device), mask], dim=1)

        for i in range(self.nb_layers):
            x = self.layers[i](x, mask=mask)

        if self.pooling == 'cls':
            x = self.out_proj(x[:, 0, :])
        else:
            x = self.out_proj(x)[:, 1:, :] # return without cls

        return x

