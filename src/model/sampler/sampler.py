import torch

def linear_schedule(t):
    return torch.clamp(t, 1e-6, 1.-1e-6)

def cosine_schedule(t):
    t = torch.cos((1-t) * torch.pi / 2)
    return torch.clamp(t, 1e-6, 1.-1e-6)


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
        if progress_bar is not None:
            r = progress_bar(r)
        for n in r:
            t_cur = self.schedule(1 -  torch.ones(b).to(x_cur.device)*(n-1)/self.n_steps )
            eps = model(x_cur, ctx, t_cur)

            x_0 = (x_cur - torch.sqrt(t_cur).reshape(b, 1, 1, 1)*eps)/torch.sqrt(1-t_cur).reshape(b, 1, 1, 1)
            x_0 = torch.clamp(x_0, -1., 1)

            # prev step
            t_cur = self.schedule(1 -  torch.ones(b).to(x_cur.device)*n/self.n_steps ).reshape(b, 1, 1, 1)
            x_cur = torch.sqrt(1-t_cur) * x_0 + torch.sqrt(t_cur)*eps

        return x_cur