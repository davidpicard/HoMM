import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.compile(mode="max-autotune")
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
                h = torch.einsum(h, mask, 'bnd, bmn -> bmd')/mask.sum(dim=2, keepdims=True) # b batch, n context tokens, m query tokens, d dim
            else:
                raise Exception('unsupported dim for mask (should be 2 or None)')
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

    def forward(self, xq, xc=None, mask=None):
        if xc is None:
            xc = xq # self attention

        # high order
        h = high_order_aggregation_(self.ho_proj(xc), self.order, mask)
        #old code not optimized through torch.compile
        # h = F.gelu(self.ho_proj(xc))
        # h = list(h.chunk(self.order, dim=-1))
        # for i in range(1, self.order):
        #     h[i] = h[i] * h[i-1]
        # h = torch.cat(h, dim=-1)
        # # averaging
        # if mask is None:
        #     h = h.mean(dim=1, keepdims=True)
        # else:
        #     if mask.dim()==2:
        #         h = (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True)/mask.unsqueeze(-1).sum(dim=1, keepdims=True)
        #     elif mask.dim() ==3:
        #         h = torch.einsum(h, mask, 'bnd, bmn -> bmd')/mask.sum(dim=2, keepdims=True) # b batch, n context tokens, m query tokens, d dim
        #     else:
        #         raise Exception('unsupported dim for mask (should be 2 or None)')

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

        self.ho_proj = nn.Linear(dim, order*order_expand*dim)
        self.se_proj = nn.Linear(dim, order*order_expand*dim)
        self.ag_proj = nn.Linear(order*order_expand*dim, dim)
        self.ho_drop = nn.Dropout(p=dropout)
        self.ffw = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(dim, ffw_expand*dim),
                                 nn.GELU(),
                                 nn.Linear(ffw_expand*dim, dim))

    def forward(self, xq, xc=None, mask=None):
        if xc is None:
            xc = xq # self attention

        # high order
        h = F.tanh(self.ho_proj(self.ho_drop(xc)))
        h = list(h.chunk(self.order, dim=-1))
        for i in range(1, self.order):
            h[i] = h[i] * h[i-1]
        h = torch.cat(h, dim=-1)
        # averaging
        if mask is None:
            h = h.mean(dim=1, keepdims=True)
        else:
            if mask.dim()==2:
                h = (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True)/mask.unsqueeze(-1).sum(dim=1, keepdims=True)
            elif mask.dim() ==3:
                h = torch.einsum(h, mask, 'bnd, bmn -> bmd')/mask.sum(dim=2, keepdims=True) # b batch, n context tokens, m query tokens, d dim
            else:
                raise Exception('unsupported dim for mask (should be 2 or None)')

        # selection
        s = F.sigmoid(self.se_proj(xq))
        sh = s * h

        # aggregation
        x = xq + self.ag_proj(sh)

        # ffw
        x = x + self.ffw(x)

        return x