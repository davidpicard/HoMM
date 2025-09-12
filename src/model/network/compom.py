import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from typing import Optional, Tuple, Dict, Any

# =============================================================================
# Core Polynomial Functions
# =============================================================================

def pom_activation(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.01, True)


def po2(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Second-order polynomial expansion.

    Args:
        x: Input tensor of shape (..., dim)

    Returns:
        Tensor of shape (..., 2*dim) with polynomial interactions
    """
    h = pom_activation(x).unsqueeze(-1)
    h2 = h * h
    h = torch.cat([h, h2], dim=-1)
    return (h * coeff).sum(-1)


def po3(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Third-order polynomial expansion.

    Args:
        x: Input tensor of shape (..., dim)

    Returns:
        Tensor of shape (..., 3*dim) with polynomial interactions
    """
    h = pom_activation(x).unsqueeze(-1)
    h2 = h * h
    h3 = h2 * h
    h = torch.cat([h, h2, h3], dim=-1)
    return (h * coeff).sum(-1)


def po4(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Fourth-order polynomial expansion.

    Args:
        x: Input tensor of shape (..., dim)

    Returns:
        Tensor of shape (..., 4*dim) with polynomial interactions
    """
    h = pom_activation(x).unsqueeze(-1)
    h2 = h * h
    h3 = h2 * h
    h4 = h2 * h2
    h = torch.cat([h, h2, h3, h4], dim=-1)
    return (h * coeff).sum(-1)


# =============================================================================
# Masking and Aggregation Functions
# =============================================================================

def mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D mask mixing for attention.

    Args:
        h: Hidden states tensor of shape (batch, seq_len, dim)
        mask: Attention mask of shape (batch, seq_len)

    Returns:
        Masked and aggregated tensor of shape (batch, 1, dim)
    """
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True) / (1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))


def full_mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply 3D mask mixing for cross-attention.

    Args:
        h: Hidden states tensor of shape (batch, seq_len, dim)
        mask: Attention mask of shape (batch, query_len, seq_len)

    Returns:
        Masked and aggregated tensor of shape (batch, query_len, dim)
    """
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h


# =============================================================================
# Polynomial Aggregation and Selection
# =============================================================================

def polynomial_aggregation_(x: torch.Tensor, coeff: torch.Tensor, k: int,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply polynomial aggregation with optional masking.

    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        coeff: Polynomial coefficients of shape TODO
        k: Polynomial order (2, 3, 4, or higher)
        mask: Optional attention mask

    Returns:
        Aggregated tensor with polynomial interactions
    """
    # Use optimized functions for common cases
    if k == 2:
        h = po2(x, coeff)
    elif k == 3:
        h = po3(x, coeff)
    elif k == 4:
        h = po4(x, coeff)
    else:
        # Generic case for k > 4
        h = pom_activation(x).unsqueeze(-1)
        h = torch.cat([h ** i for i in range(k)], dim=-1)  # TODO vectorize
        h = (h * coeff).sum(-1)

    # Apply masking if provided
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim() == 2:
            h = mask_mixer(h, mask.to(h.device))
        elif mask.dim() == 3:
            h = full_mask_mixer(h, mask.to(h.device))
        else:
            raise ValueError(f'Unsupported mask dimension: {mask.dim()}. Expected 2, 3, or None.')
    return h


def polynomial_selection_(x: torch.Tensor, h: torch.Tensor, n_sel_heads: int) -> torch.Tensor:
    """
    Apply polynomial selection with sigmoid gating.

    Args:
        x: Query tensor
        h: Context tensor from polynomial aggregation

    Returns:
        Gated output tensor
    """
    s = F.sigmoid(x).unsqueeze(-1)
    b, n, c, _ = s.shape
    bb, nn, d = h.shape
    h = h.view(bb, nn, c, -1)
    # print(f"s: {s.shape} h: {h.shape}-{nn, d} s*h: {(s*h).shape}")
    return (s * h).view(b, n, d)


# =============================================================================
# Main PoM Function
# =============================================================================

def pom(xq: torch.Tensor, xc: torch.Tensor, coeff: torch.Tensor, k: int, n_sel_heads: int,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Polynomial Mixer (PoM) operation.

    This function implements the polynomial mixer operation which combines
    polynomial aggregation of context with selection from queries.

    Args:
        xq: Query input tensor of shape (batch, query_len, dim)
        xc: Context input tensor of shape (batch, context_len, dim)
        coeff: Polynomial coefficients of shape TODO
        k: Polynomial order (degree of interactions to capture)
        mask: Optional attention mask for masking specific positions

    Returns:
        Output tensor after polynomial mixing
    """
    h = polynomial_aggregation_(xc, coeff, k, mask)
    o = polynomial_selection_(xq, h, n_sel_heads)
    return o


# =============================================================================
# ComPoM Module Class
# =============================================================================

class ComPoM(nn.Module):
    """
    More compact Polynomial Mixer (PoM) Module.

    A custom neural network layer designed for capturing higher-order interactions
    between input features through polynomial expansions. This module consists of
    three linear projections and a custom PoM operation.

    Attributes:
        dim (int): The dimensionality of the input features
        order (int): The order of the polynomial interactions to capture
        order_expand (int): The expansion factor for the polynomial order
        po_proj (nn.Linear): Linear projection for polynomial computation
        se_proj (nn.Linear): Linear projection for selection mechanism
        ag_proj (nn.Linear): Linear projection for output aggregation
        pom (callable): The polynomial mixer operation function
    """

    def __init__(self, dim: int, degree: int, expand: int, n_groups: int, n_sel_heads: int, bias: bool = False):
        """
        Initialize the PoM module.

        Args:
            dim: The dimensionality of the input features
            degree: The degree of the polynomial to capture
            expand: The expansion factor for the polynomial order
            bias: Whether to include bias terms in linear projections
        """
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand
        self.n_groups = n_groups
        self.n_sel_heads = n_sel_heads
        assert dim % n_groups == 0, "dim must be divisible by n_groups for group conv"
        assert dim * expand % n_sel_heads == 0, "dim * expand must be divisible by n_sel_heads"

        # Linear projections
        if self.n_groups > 1:
            self.po_proj = nn.Conv1d(dim, expand * dim, kernel_size=1, bias=bias, groups=n_groups)
        else:
            self.po_proj = nn.Linear(dim, expand * dim, bias=bias)
        self.po_coeff = nn.Parameter((2.*torch.randn(dim * expand, degree)).clamp(-2., 2.))
        self.se_proj = nn.Linear(dim, n_sel_heads, bias=bias)
        self.ag_proj = nn.Linear(expand * dim, dim, bias=bias)
        self.pom = pom

    def forward(self, xq: torch.Tensor, xc: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the PoM module.

        Args:
            xq: Query input tensor of shape (batch, n_tokens, dim)
            xc: Context input tensor. If None, self-attention is performed
            mask: Optional attention mask tensor

        Returns:
            Output tensor after applying the PoM operation
        """
        if xc is None:
            xc = xq  # self-attention

        s = self.se_proj(xq)
        if self.n_groups > 1:
            h = self.po_proj(xc.transpose(1, 2)).transpose(1, 2)
        else:
            h = self.po_proj(xc)
        sh = self.pom(s, h, self.po_coeff, self.order, self.n_sel_heads, mask)

        return self.ag_proj(sh)

    def state_forward(self, xq: torch.Tensor, xc: Optional[torch.Tensor] = None,
                      state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with state management for incremental processing.

        Args:
            xq: Query input tensor
            xc: Context input tensor. If None, self-attention is performed
            state: Optional state dictionary from previous forward pass

        Returns:
            Tuple of (output_tensor, new_state)
        """
        if xc is None:
            xc = xq  # self-attention

        s = self.se_proj(xq)
        xc = self.po_proj(xc)
        h_current = polynomial_aggregation_(xc, self.po_coeff, self.order)
        n_current = h_current.shape[1]

        if state is not None:
            h_past = state['h']
            n_past = state['n']
            h = (n_past * h_past + n_current * h_current) / (n_past + n_current)
        else:
            h = h_current
            n_past = 0

        new_state = {'h': h, 'n': n_past + n_current}

        sh = polynomial_selection_(s, h, self.n_sel_heads)
        return self.ag_proj(sh), new_state