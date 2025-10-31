import torch



class Rotary(torch.nn.Module):
    """Rotary position embeddings."""

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        if (not self.seq_len_cached) or seq_len >= self.seq_len_cached:
            # print(f"rotary.forward caching rotary: {seq_len} != {self.seq_len_cached}")
            self.seq_len_cached = seq_len
            freqs = torch.outer(t.type_as(self.inv_freq), self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, t, None, :], self.sin_cached[None, t, None, :]

    def position_forward(self, pos, seq_len, device="cpu"):
        if seq_len != self.seq_len_cached:
            # print(f"rotary.ar_forward caching rotary: {seq_len} != {self.seq_len_cached}")
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, pos, None, :], self.sin_cached[None, pos, None, :]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
