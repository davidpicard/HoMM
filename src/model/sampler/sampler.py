import torch

def linear_schedule(t):
    return torch.clamp(t, 1e-6, 1.-1e-6)

def cosine_schedule(t):
    t = torch.cos((1-t) * torch.pi / 2)
    return torch.clamp(t, 1e-6, 1.-1e-6)

# taken from N Dufour
class SigmoidScheduler:
    def __init__(self, start=-3, end=3, tau=1, clip_min=1e-9):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min

        self.v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
        self.v_end = torch.sigmoid(torch.tensor(self.end / self.tau))

    def __call__(self, t):
        output = (
            -torch.sigmoid((t * (self.end - self.start) + self.start) / self.tau)
            + self.v_end
        ) / (self.v_end - self.v_start)
        return torch.clamp(output, min=self.clip_min, max=1.0)

class DDIM():
    def __init__(
            self,
            n_steps,
            schedule,
    ):
        self.n_steps = n_steps
        self.schedule = SigmoidScheduler()

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

            # prev step
            t_cur = self.schedule((n-1)/self.n_steps  * torch.ones(b).to(x_cur.device)).reshape(b, 1, 1, 1)
            x_cur = torch.sqrt(1-t_cur) * x_0 + torch.sqrt(t_cur)*eps

        return x_cur