import einops
import torch

def linear_schedule(t):
    return torch.clamp(t, 1e-6, 1.-1e-6)

def sine_schedule(t):
    t = torch.sin(t * torch.pi / 2)
    return torch.clamp(t, 1e-6, 1.-1e-6)

def cosine_schedule(t):
    t = 1 - torch.cos(t * torch.pi / 2)
    return torch.clamp(t, 1e-6, 1.-1e-6)

def sigmoid_schedule(t):
    t = torch.sigmoid(-2. + 4*t)
    o = torch.tensor(1)
    t = (t-torch.sigmoid(-2.*o))/(torch.sigmoid(2.*o)-torch.sigmoid(-2.*o))
    return torch.clamp(t, 1e-6, 1.-1e-6)
    # start = -1
    # end = 3
    # tau = 0.5
    # v_start = torch.sigmoid(torch.tensor(start/tau))
    # v_end = torch.sigmoid(torch.tensor(end/tau))
    # return 1 - (v_end - torch.sigmoid((t*(end-start) + start))/tau)/(v_end - v_start)

def karras_schedule(t):
    sigma_min = 1e-2
    sigma_max = 2
    rho = 7.
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    t = 1-(max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
    return torch.clamp(t, 1e-6, 1.-1e-6)

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
                 clip_img_pred=False,
                 clip_value = 1.):
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
            x_0 = x_0.clamp(-self.clip_value, self.clip_value)
            noise_pred = (samples - torch.sqrt(1-sigma) * x_0) / torch.sqrt(sigma)
        # recompute sample at previous step
        t = t - self.train_timesteps/self.num_inference_steps
        sigma = self.schedule(t.clamp(0, self.train_timesteps-1)/self.train_timesteps).view(b, 1, 1, 1)
        samples = torch.sqrt(1 - sigma) * x_0 + torch.sqrt(sigma) * noise_pred
        # samples.clamp(-1, 1.)
        return samples, x_0

class DDPMLinearScheduler():
    def __init__(self,
                 n_timesteps,
                 schedule = linear_schedule,
                 clip_img_pred=False,
                 clip_value = 1.):
        self.train_timesteps = n_timesteps
        self.timesteps = None
        self.schedule = schedule
        self.clip_img_pred = clip_img_pred
        self.clip_value = clip_value

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
        sigma_now = self.schedule(t.clamp(0, self.train_timesteps - 1) / self.train_timesteps).view(b, 1, 1, 1)
        x_pred = (samples - torch.sqrt(sigma_now) * noise_pred) / torch.sqrt(1-sigma_now)
        if self.clip_img_pred:
            x_pred = torch.clamp(x_pred, -self.clip_value, self.clip_value)
            noise_est = (samples - torch.sqrt(1-sigma_now) * x_pred) / torch.sqrt(sigma_now)
            # noise_est = noise_pred
        else:
            noise_est = noise_pred

        t = t - self.train_timesteps/self.num_inference_steps
        sigma_next = self.schedule(t.clamp(0, self.train_timesteps-1)/self.train_timesteps).view(b, 1, 1, 1)
        log_alpha_t = torch.log(1-sigma_now) - torch.log(1-sigma_next)
        alpha_t = torch.clip(torch.exp(log_alpha_t), 0, 1)
        x_mean = torch.rsqrt(alpha_t) * (
            samples - torch.rsqrt(sigma_now) * (1 - alpha_t) * noise_est
        )
        var_t = 1 - alpha_t
        eps = torch.randn(samples.shape, device=samples.device)
        x_next = x_mean + torch.sqrt(var_t) * eps
        return x_next, x_pred



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

    @torch.no_grad()
    def sample_cfg(self, samples, class_labels, cfg, device, num_inference_steps: int = 50, step_callback = None, cfg_scheduler=None):
        batch_size = len(class_labels)
        class_labels = class_labels.to(device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in (self.scheduler.timesteps):
            timesteps = t
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(batch_size).to(device)
            # duplicate all
            x_input = torch.cat([samples, samples], dim=0).to(device)
            t_input = torch.cat([timesteps, timesteps], dim=0).to(device)
            c_input = torch.cat([class_labels, class_labels], dim=0).to(device)
            c_input[batch_size:] = 1000
            # predict noise model_output
            noise_pred = self.model(x_input, time=t_input, cls=c_input)
            eps_c = noise_pred[0:batch_size, ...]
            eps_u = noise_pred[batch_size:, ...]
            w = 1.
            if cfg_scheduler is not None:
                w = cfg_scheduler(t/self.scheduler.train_timesteps)
            eps = eps_c + w*cfg*(eps_c - eps_u)
            # compute previous image: x_t -> x_t-1
            samples, x_0 = self.scheduler.step(eps, timesteps, samples)
            if step_callback is not None:
                step_callback(t, samples, x_0, noise_pred)

        # samples = (samples / 2 + 0.5).clamp(0, 1)
        return samples

##  cfg sched

def linear(t):
    return 1-t
def clamp_linear(c=0.1):
    return lambda t: torch.clamp_min_(1-t, c)
def trunc_linear(c=0.1):
    return lambda t: (1-t)*((1-t)>c)

