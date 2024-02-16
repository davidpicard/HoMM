import einops
import torch

def linear_schedule(t):
    return torch.clamp(t, 1e-6, 1.-1e-6)

def cosine_schedule(t):
    t = torch.cos((1-t) * torch.pi / 2)
    return torch.clamp(t, 1e-6, 1.-1e-6)

def sigmoid_schedule(t):
    return torch.sigmoid(-3 + 6*t).clamp(1e-6, 1.-1e-6)
    # start = -1
    # end = 3
    # tau = 0.5
    # v_start = torch.sigmoid(torch.tensor(start/tau))
    # v_end = torch.sigmoid(torch.tensor(end/tau))
    # return 1 - (v_end - torch.sigmoid((t*(end-start) + start))/tau)/(v_end - v_start)

# adapted from N Dufour
class SigmoidScheduler:
    def __init__(self, start=-1, end=3, tau=0.5, clip_min=1e-9):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min

        self.v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
        self.v_end = torch.sigmoid(torch.tensor(self.end / self.tau))

    def __call__(self, t):
        output = 1-(
            -torch.sigmoid((t * (self.end - self.start) + self.start) / self.tau)
            + self.v_end
        ) / (self.v_end - self.v_start)
        return torch.clamp(output, min=self.clip_min, max=1.0)



class DDIMLinearScheduler():
    def __init__(self,
                 n_timesteps,
                 schedule = linear_schedule,
                 clip_img_pred=False):
        self.train_timesteps = n_timesteps
        self.timesteps = None
        self.schedule = schedule
        self.clip_img_pred = clip_img_pred

    def add_noise(self, x, noise, t):
        t = torch.clamp(t, 0, self.train_timesteps)
        sigma = self.schedule(t/self.train_timesteps).view(x.shape[0], 1, 1, 1)
        return torch.sqrt(1-sigma)*x + torch.sqrt(sigma)*noise

    def set_timesteps(self, num_inference_steps):
        timesteps = torch.linspace(0, self.train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps.flip(0)

    def step(self, noise_pred, t, samples):
        b, c, h, w = samples.shape
        # pred x0
        sigma = self.schedule(t.clamp(0, self.train_timesteps-1)/self.train_timesteps).view(b, 1, 1, 1)
        x_0 = (samples - torch.sqrt(sigma) * noise_pred) / torch.sqrt(1 - sigma)
        if self.clip_img_pred:
            x_0 = x_0.clamp(-1, 1)
        # recompute sample at previous step
        t = t - self.train_timesteps/self.num_inference_steps
        sigma = self.schedule(t.clamp(0, self.train_timesteps-1)/self.train_timesteps).view(b, 1, 1, 1)
        samples = torch.sqrt(1 - sigma) * x_0 + torch.sqrt(sigma) * noise_pred
        # samples.clamp(-1, 1.)
        return samples, x_0

class DiTPipeline():
    def __init__(
            self,
            model,
            scheduler,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler


    @torch.no_grad()
    def __call__(
        self,
        samples,
        class_labels,
        device,
        num_inference_steps: int = 50,
        step_callback=None,
    ):

        batch_size = len(class_labels)
        im_size = self.model.im_size

        class_labels = class_labels.to(device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in (self.scheduler.timesteps):
            timesteps = t
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(batch_size).to(device)
            # predict noise model_output
            noise_pred = self.model(samples, time=timesteps, cls=class_labels)
            # compute previous image: x_t -> x_t-1
            samples, x_0 = self.scheduler.step(noise_pred, timesteps, samples)
            if step_callback is not None:
                step_callback(t, samples, x_0, noise_pred)

        # samples = (samples / 2 + 0.5).clamp(0, 1)
        return samples
