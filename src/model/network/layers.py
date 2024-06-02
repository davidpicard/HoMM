import torch
import torch.nn as nn
import torch.nn.functional as F


def high_order_aggregation_(x: torch.Tensor, k: int, mask=None):
        h = list(F.gelu(x).chunk(k, dim=-1))
        for i in range(1, k):
            h[i] = h[i] * h[i-1]
        h = torch.cat(h, dim=-1)
        if mask is None:
            h = h.mean(dim=1, keepdims=True)
        else:
            if mask.dim()==2:
                h = (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True)/mask.unsqueeze(-1).sum(dim=1, keepdims=True)
            elif mask.dim() ==3:
                mask = mask.type(h.dtype)
                h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
                h = h/(1+mask.sum(dim=2, keepdims=True))
            else:
                raise Exception('unsupported dim for mask (should be 2,3 or None)')
        return h


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
        self.high_order_aggregation_ = high_order_aggregation_

    def forward(self, xq, xc=None, mask=None):
        if xc is None:
            xc = xq # self attention

        # high order
        h = self.high_order_aggregation_(self.ho_proj(xc), self.order, mask)

        # selection
        s = F.sigmoid(self.se_proj(xq))
        sh = s * h

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