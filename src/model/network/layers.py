import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True

@torch.compile
def gelu(x: torch.Tensor):
    # return x * torch.erf(x)
    return F.gelu(x)

@torch.compile
def po2(x: torch.Tensor):
    h1, h2 = gelu(x).chunk(2, dim=-1)
    h2 = h2 * h1
    return torch.cat([h1, h2], dim=-1)

@torch.compile
def po3(x: torch.Tensor):
    h1, h2, h3 = gelu(x).chunk(3, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    return torch.cat([h1, h2, h3], dim=-1)

@torch.compile
def po4(x: torch.Tensor):
    h1, h2, h3, h4 = gelu(x).chunk(4, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    h4 = h4 * h3
    return torch.cat([h1, h2, h3, h4], dim=-1)


def mask_mixer(h, mask):
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True)/(1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))


def full_mask_mixer(h, mask):
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h

def high_order_aggregation_(x: torch.Tensor, k: int, mask=None):
    if k == 2:
        h = po2(x)
    elif k == 3:
        h = po3(x)
    elif k == 4:
        h = po4(x)
    else:
        h = list(gelu(x).chunk(k, dim=-1))
        for i in range(1, k):
            h[i] = h[i] * h[i-1]
        h = torch.cat(h, dim=-1)
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim()==2:
            h = mask_mixer(h, mask)
        elif mask.dim() ==3:
            h = full_mask_mixer(h, mask)
        else:
            raise Exception('unsupported dim for mask (should be 2,3 or None)')
    return h

@torch.compile
def high_order_selection_(x: torch.Tensor, h: torch.Tensor):
    return F.sigmoid(x) * h

def hom(xq: torch.Tensor, xc: torch.Tensor, k: int, mask=None):
    h = high_order_aggregation_(xc, k, mask)
    o = high_order_selection_(xq, h)
    return o

class HoM(nn.Module):
    def __init__(self, dim, order, order_expand, bias=True):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand

        self.ho_proj = nn.Linear(dim, order*order_expand*dim, bias=bias)
        self.se_proj = nn.Linear(dim, order*order_expand*dim, bias=bias)
        self.ag_proj = nn.Linear(order*order_expand*dim, dim, bias=bias)

        #self.high_order_aggregation_ = torch.compile(high_order_aggregation_, mode="max-autotune", fullgraph=False)
        # self.high_order_aggregation_ = high_order_aggregation_
        # self.high_order_selection_ = high_order_selection_
        self.hom = hom

    def forward(self, xq, xc=None, mask=None):
        if xc is None:
            xc = xq # self attention

        s = self.se_proj(xq)
        h = self.ho_proj(xc)
        sh = self.hom(s, h, self.order, mask)

        # aggregation
        return self.ag_proj(sh)


class HoMLayer(nn.Module):
    def __init__(self, dim, order, order_expand, ffw_expand, dropout=0.):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.hom = HoM(self.dim, self.order, self.order_expand)
        self.ffw = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(dim, ffw_expand*dim),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand*dim, dim))

    def forward(self, xq, xc=None, mask=None):
        if xc is None:
            xc = xq # self attention

        # high order
        x = xq + self.hom(xq, xc, mask)

        # ffw
        x = x + self.ffw(x)

        return x