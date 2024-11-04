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
                 model,
                 schedule = linear_schedule,
                 clip_img_pred=False,
                 clip_value = 1.):
        self.model = model
        self.train_timesteps = model.n_timesteps
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

    def step(self, i, samples, ctx, cfg=0.):
        b, d, h, w = samples.shape
        t = self.timesteps[i] * torch.ones(b, 1, 1, 1).to(samples.device)
        # predict noise model_output
        if cfg > 0:
            # duplicate all
            x_input = torch.cat([samples, samples], dim=0)
            t_input = torch.cat([t, t], dim=0).to(samples.device)
            c_input = torch.cat([ctx, ctx], dim=0).to(samples.device)
            c_input[b:] = 1000
            noise_pred = self.model(x_input, time=t_input.squeeze(), cls=c_input)
            eps_c = noise_pred[0:b, ...]
            eps_u = noise_pred[b:, ...]
            noise_pred = eps_c + cfg*(eps_c - eps_u)
        else:
            noise_pred = self.model(samples, time=t.squeeze(), cls=ctx)
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

    @torch.no_grad()
    def sample(self, samples, class_labels, cfg: float = 0., num_inference_steps: int = 50, step_callback = None, cfg_scheduler=None):
        batch_size = len(class_labels)
        class_labels = class_labels.to(samples.device)

        # set step values
        self.set_timesteps(num_inference_steps)
        for i in range(self.num_inference_steps):
            # compute previous image: x_t -> x_t-1
            samples, x_0 = self.step(i, samples, class_labels, cfg=cfg)
            if step_callback is not None:
                step_callback(i, samples, x_0)
        return samples

class DDPMLinearScheduler():
    def __init__(self,
                 model,
                 schedule = linear_schedule,
                 clip_img_pred=False,
                 clip_value = 1.,
                 clip_timestep = 0.):
        self.model = model
        self.train_timesteps = model.n_timesteps
        self.timesteps = None
        self.schedule = schedule
        self.clip_img_pred = clip_img_pred
        self.clip_value = clip_value
        self.clip_timestep = clip_timestep*self.model.n_timesteps

    def add_noise(self, x, noise, t):
        t = torch.clamp(t, 0, self.train_timesteps)
        sigma = self.schedule(t/self.train_timesteps).view(x.shape[0], 1, 1, 1)
        return torch.sqrt(1-sigma)*x + torch.sqrt(sigma)*noise

    def set_timesteps(self, num_inference_steps):
        timesteps = torch.linspace(0, self.train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps.flip(0)

    def step(self, i, samples, ctx, cfg=0.):
        b, d, h, w = samples.shape
        t = self.timesteps[i] * torch.ones(b, 1, 1, 1).to(samples.device)
        # predict noise model_output
        if cfg > 0:
            # duplicate all
            x_input = torch.cat([samples, samples], dim=0)
            t_input = torch.cat([t, t], dim=0).to(samples.device)
            c_input = torch.cat([ctx, ctx], dim=0).to(samples.device)
            c_input[b:] = 1000
            noise_pred = self.model(x_input, time=t_input.squeeze(), cls=c_input)
            eps_c = noise_pred[0:b, ...]
            eps_u = noise_pred[b:, ...]
            noise_pred = eps_c + cfg*(eps_c - eps_u)
        else:
            noise_pred = self.model(samples, time=t.squeeze(), cls=ctx)
        sigma_now = self.schedule(t.clamp(0, self.train_timesteps - 1) / self.train_timesteps).view(b, 1, 1, 1)
        x_pred = (samples - torch.sqrt(sigma_now) * noise_pred) / torch.sqrt(1-sigma_now)
        if self.clip_img_pred and t[0] > self.clip_timestep:
            x_pred = torch.clamp(x_pred, -self.clip_value, self.clip_value)
            noise_est = (samples - torch.sqrt(1-sigma_now) * x_pred) / torch.sqrt(sigma_now)
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

    @torch.no_grad()
    def sample(self, samples, class_labels, cfg: float = 0., num_inference_steps: int = 50, step_callback = None, cfg_scheduler=None):
        batch_size = len(class_labels)
        class_labels = class_labels.to(samples.device)

        # set step values
        self.set_timesteps(num_inference_steps)
        for i in range(self.num_inference_steps):
            # compute previous image: x_t -> x_t-1
            samples, x_0 = self.step(i, samples, class_labels, cfg=cfg)
            if step_callback is not None:
                step_callback(i, samples, x_0)
        return samples


class DPMScheduler():
    def __init__(self,
                 model,
                 clip_img_pred=True,
                 clip_value=1.5):
        self.model = model
        self.train_timesteps = model.n_timesteps
        self.timesteps = None
        self.clip_img_pred = clip_img_pred
        self.clip_value = clip_value
        self.noise_prev = None

    def add_noise(self, x, noise, t):
        t = torch.clamp(t, 0, self.train_timesteps)
        sigma = (t / self.train_timesteps).view(x.shape[0], 1, 1, 1)
        return torch.sqrt(1 - sigma) * x + torch.sqrt(sigma) * noise

    def set_timesteps(self, num_inference_steps):
        timesteps = torch.linspace(1.0, self.train_timesteps - 1, num_inference_steps + 1)
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps.flip(0)
        self.noise_prev = None

    def _predict_noise(self, samples, t, ctx, cfg=0.):
        b, d, h, w = samples.shape
        if cfg > 0:
            # duplicate all
            x_input = torch.cat([samples, samples], dim=0)
            t_input = torch.cat([t, t], dim=0).to(samples.device)
            c_input = torch.cat([ctx, ctx], dim=0).to(samples.device)
            c_input[b:] = 1000
            noise_pred = self.model(x_input, time=t_input.squeeze(), cls=c_input)
            eps_c = noise_pred[0:b, ...]
            eps_u = noise_pred[b:, ...]
            noise_pred = eps_c + cfg * (eps_c - eps_u)
        else:
            noise_pred = self.model(samples, time=t.squeeze(), cls=ctx)

        if self.clip_img_pred:
            sigma_now = torch.sqrt(t / self.train_timesteps)
            alpha_now = torch.sqrt(1 - t / self.train_timesteps)
            x_pred = (samples - sigma_now * noise_pred) / alpha_now
            x_pred = torch.clamp(x_pred, -self.clip_value, self.clip_value)
            noise_pred = (samples - alpha_now * x_pred) / sigma_now
        return noise_pred

    def _step1(self, i, samples, ctx, cfg=0.):
        b, d, h, w = samples.shape
        t_now = self.timesteps[i] * torch.ones(b, 1, 1, 1).to(samples.device)

        sigma_now = torch.sqrt(t_now / self.train_timesteps)
        alpha_now = torch.sqrt(1 - t_now / self.train_timesteps)
        lambda_now = torch.log(alpha_now / sigma_now)

        noise_est = self._predict_noise(samples, t_now, ctx, cfg)
        x_pred = (samples - sigma_now * noise_est) / alpha_now

        t_next = self.timesteps[i + 1] * torch.ones(b, 1, 1, 1).to(samples.device) / self.train_timesteps
        sigma_next = torch.sqrt(t_next)
        alpha_next = torch.sqrt(1 - t_next)
        lambda_next = torch.log(alpha_next / sigma_next)
        h = lambda_next - lambda_now

        x_next = alpha_next / alpha_now * samples - sigma_now * torch.expm1(h) * noise_est

        self.noise_prev = noise_est

        #         print(f"tn: {t_now[0].item()} an: {alpha_now[0].item()} sn: {sigma_now[0].item()} ln: {lambda_now[0].item()} tx: {t_next[0].item()} ax: {alpha_next[0].item()} sx: {sigma_next[0].item()} lx: {lambda_next[0].item()} h: {h[0].item()}")

        return x_next, x_pred

    def _step2(self, i, samples, ctx, cfg=0.):
        b, d, h, w = samples.shape

        t_now = self.timesteps[i] * torch.ones(b, 1, 1, 1).to(samples.device)
        sigma_now = torch.sqrt(t_now / self.train_timesteps)
        alpha_now = torch.sqrt(1 - t_now / self.train_timesteps)
        lambda_now = torch.log(alpha_now / sigma_now)

        t_prev = self.timesteps[i - 1] * torch.ones(b, 1, 1, 1).to(samples.device)
        sigma_prev = torch.sqrt(t_prev / self.train_timesteps)
        alpha_prev = torch.sqrt(1 - t_prev / self.train_timesteps)
        lambda_prev = torch.log(alpha_prev / sigma_prev)

        t_next = self.timesteps[i + 1] * torch.ones(b, 1, 1, 1).to(samples.device)
        sigma_next = torch.sqrt(t_next / self.train_timesteps)
        alpha_next = torch.sqrt(1 - t_next / self.train_timesteps)
        lambda_next = torch.log(alpha_next / sigma_next)

        noise_prev = self.noise_prev
        noise_now = self._predict_noise(samples, t_now, ctx, cfg)

        h = lambda_next - lambda_now
        h_prev = lambda_now - lambda_prev
        r0 = h_prev / h

        D0 = noise_now
        D1 = 1. / r0 * (noise_now - noise_prev)

        x_next = alpha_next / alpha_now * samples - sigma_next * torch.expm1(h) * D0 \
                 - 0.5 * sigma_next * torch.expm1(h) * D1
        x_pred = (samples - sigma_now * noise_now) / alpha_now

        self.noise_prev = noise_now
        return x_next, x_pred

    def step(self, i, samples, ctx, cfg=0.):
        b, c, h, w = samples.shape
        if self.noise_prev is None:
            return self._step1(i, samples, ctx, cfg)
        return self._step2(i, samples, ctx, cfg)

    @torch.no_grad()
    def sample(self, samples, class_labels, cfg: float = 0., num_inference_steps: int = 50, step_callback=None,
               cfg_scheduler=None):
        batch_size = len(class_labels)
        class_labels = class_labels.to(samples.device)

        # set step values
        self.set_timesteps(num_inference_steps)
        for i in range(self.num_inference_steps):
            # compute previous image: x_t -> x_t-1
            samples, x_0 = self.step(i, samples, class_labels, cfg=cfg)
            if step_callback is not None:
                step_callback(i, samples, x_0)
        return samples


import math
class FlowMatchingSampler():
    def __init__(self,
                 model,
                 size=32):
        self.model = model
        self.train_timesteps = model.n_timesteps
        self.timesteps = None
        self.size = size
        print(f"Flow matching sampler with size {self.size}")

    def rescale_t(self, t):
        t = t/self.train_timesteps
        ts = t * math.sqrt(self.size / 32) / (1 + (math.sqrt(self.size / 32) - 1) * t)
        return ts * self.train_timesteps

    def add_noise(self, x, noise, t):
        t = torch.clamp(t, 0, self.train_timesteps)
        ts = self.rescale_t(t)/ self.train_timesteps
        sigma = ts.view(x.shape[0], 1, 1, 1)
        alpha = 1 - sigma
        return alpha * x + sigma * noise

    def set_timesteps(self, num_inference_steps):
        timesteps = torch.linspace(1.0, self.train_timesteps - 1, num_inference_steps + 1)
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps.flip(0)
        self.noise_prev = None

    def _predict(self, samples, t, ctx, cfg=0.):
        b, d, h, w = samples.shape
        if cfg > 0:
            # duplicate all
            x_input = torch.cat([samples, samples], dim=0)
            t_input = torch.cat([t, t], dim=0).to(samples.device)
            c_input = torch.cat([ctx, ctx], dim=0).to(samples.device)
            c_input[b:] = 1000
            pred = self.model(x_input, time=t_input.squeeze(), cls=c_input)
            eps_c = pred[0:b, ...]
            eps_u = pred[b:, ...]
            pred = eps_c + cfg * (eps_c - eps_u)
        else:
            pred = self.model(samples, time=t.squeeze(), cls=ctx)

        return pred

    @torch.no_grad()
    def sample(self, samples, class_labels, cfg: float = 0., num_inference_steps: int = 50, step_callback=None, cfg_scheduler = None):
        class_labels = class_labels.to(samples.device)
        b, c, h, w = samples.shape
        # set step values
        self.set_timesteps(num_inference_steps)
        for i, t in enumerate(zip(self.timesteps[0:-2], self.timesteps[1:])):
            tc, tn = t
            tc = self.rescale_t(tc) * torch.ones((b, 1, 1, 1)).to(samples.device)
            tn = self.rescale_t(tn) * torch.ones((b, 1, 1, 1)).to(samples.device)
            # compute previous image: x_t -> x_t-1
            dt = (tc-tn) / self.train_timesteps
            samples = samples - dt*self._predict(samples, tc, class_labels, cfg)
            if step_callback is not None:
                step_callback(i, samples, samples)
        return samples

class HeunVelocitySampler():
    def __init__(self,
                 model):
        self.model = model
        self.train_timesteps = model.n_timesteps
        self.timesteps = None
        self.size = model.im_size

    def rescale_t(self, t):
        t = t/self.train_timesteps
        ts = t * math.sqrt(self.size / 32) / (1 + (math.sqrt(self.size / 32) - 1) * t)
        return ts * self.train_timesteps

    def add_noise(self, x, noise, t):
        t = torch.clamp(t, 0, self.train_timesteps)
        t = self.rescale_t(t)
        sigma = (t / self.train_timesteps).view(x.shape[0], 1, 1, 1)
        return (1 - sigma) * x + sigma * noise

    def set_timesteps(self, num_inference_steps):
        timesteps = torch.linspace(1.0, self.train_timesteps - 1, num_inference_steps + 1)
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps.flip(0)
        self.noise_prev = None

    def _predict(self, samples, t, ctx, cfg=0.):
        b, d, h, w = samples.shape
        if cfg > 0:
            # duplicate all
            x_input = torch.cat([samples, samples], dim=0)
            t_input = torch.cat([t, t], dim=0).to(samples.device)
            c_input = torch.cat([ctx, ctx], dim=0).to(samples.device)
            c_input[b:] = 1000
            pred = self.model(x_input, time=t_input.squeeze(), cls=c_input)
            eps_c = pred[0:b, ...]
            eps_u = pred[b:, ...]
            pred = eps_c + cfg * (eps_c - eps_u)
        else:
            pred = self.model(samples, time=t.squeeze(), cls=ctx)

        return pred

    @torch.no_grad()
    def sample(self, samples, class_labels, cfg: float = 0., num_inference_steps: int = 50, step_callback=None, cfg_scheduler = None):
        class_labels = class_labels.to(samples.device)
        b, c, h, w = samples.shape
        # set step values
        self.set_timesteps(num_inference_steps)
        for i, (t, tp1) in enumerate(zip(self.timesteps[0:-1], self.timesteps[1:])):
            t = self.rescale_t(t)
            tp1 = self.rescale_t(tp1)
            t = t * torch.ones((b, 1, 1, 1)).to(samples.device)
            tp1 = tp1 * torch.ones((b, 1, 1, 1)).to(samples.device)
            dt = (t - tp1)/self.train_timesteps

            di = self._predict(samples, t, class_labels, cfg)
            xi = samples - dt * di

            dip1 = self._predict(xi, tp1, class_labels, cfg)
            samples = samples - dt*(di + dip1)/2

            if step_callback is not None:
                step_callback(i, samples, samples)
        return samples


##  cfg sched

def linear(t):
    return 1-t
def clamp_linear(c=0.1):
    return lambda t: torch.clamp_min_(1-t, c)
def trunc_linear(c=0.1):
    return lambda t: (1-t)*((1-t)>c)

