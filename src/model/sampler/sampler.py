import torch

def linear_schedule(t):
    return torch.clamp(t, 1e-6, 1.-1e-6)

def cosine_schedule(t):
    t = torch.cos((1-t) * torch.pi / 2)
    return torch.clamp(t, 1e-6, 1.-1e-6)


import matplotlib.pyplot as plt

from ..diffusion import denormalize

class DDIM():
    def __init__(
            self,
            n_steps,
            schedule,
    ):
        self.n_steps = n_steps
        self.schedule = linear_schedule

    @torch.no_grad()
    def sample(self, noise, model, ctx, progress_bar=None):
        x_cur = noise
        b, c, h, w = x_cur.shape
        r = torch.arange(1, self.n_steps+1)
        r = self.n_steps+1 - r
        if progress_bar is not None:
            r = progress_bar(r)
        for n in r:
            t_cur = self.schedule(n/self.n_steps ) * torch.ones(b).to(x_cur.device)
            eps = model(x_cur, ctx, t_cur)

            x_0 = (x_cur - torch.sqrt(t_cur).reshape(b, 1, 1, 1)*eps)/torch.sqrt(1-t_cur).reshape(b, 1, 1, 1)
            x_0 = torch.clamp(x_0, -1., 1)

            if n%25 == 0:
                samples = denormalize(x_0).detach().cpu()
                samples = einops.rearrange(samples, "(b m) c h w -> (b h) (m w) c", b=2)
                plt.imshow(samples)
                plt.show()

            # prev step
            t_cur = self.schedule((n-1)/self.n_steps  * torch.ones(b).to(x_cur.device)).reshape(b, 1, 1, 1)
            x_cur = torch.sqrt(1-t_cur) * x_0 + torch.sqrt(t_cur)*eps

        return x_cur