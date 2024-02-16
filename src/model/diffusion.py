import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torchvision.transforms import transforms

from .sampler.sampler import DiTPipeline, DDIMLinearScheduler

denormalize = transforms.Normalize(
    mean=[-1],
    std=[2.],
)
class DiffusionModule(L.LightningModule):
    def __init__(
            self,
            model,
            mode,
            loss,
            optimizer_cfg,
            lr_scheduler_builder,
            # train_batch_preprocess,
            val_sampler,
            torch_compile=False,
            latent_vae=False
        ):
        super().__init__()
        # do optim
        if torch_compile:
            print("compiling model")
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        self.model = model
        self.mode = mode
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder
        # self.train_batch_preprocess = train_batch_preprocess
        self.val_sampler = val_sampler
        self.latent_vae = latent_vae
        if latent_vae:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", device="cuda:0", subfolder="vae", use_safetensors=True)
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        # noise scheduler
        self.n_timesteps = model.n_timesteps
        self.scheduler = DDIMLinearScheduler(n_timesteps=self.n_timesteps)
        self.pipeline = DiTPipeline(model=model, scheduler=self.scheduler)

    def vae_encode(self, x):
        if self.latent_vae:
            x = self.vae.encode(x).latent_dist.sample()
            x =  x * self.vae.config.scaling_factor
            # x = x - torch.tensor([[5.81, 3.25, 0.12, -2.15]]).unsqueeze(-1).unsqueeze(-1).to(x.device)
            # x = x * 0.5 / torch.tensor([[4.17, 4.62, 3.71, 3.28]]).unsqueeze(-1).unsqueeze(-1).to(x.device)
        return x

    def vae_decode(self, x):
        if self.latent_vae:
            x =  x / self.vae.config.scaling_factor
            # x = x / 0.5 * torch.tensor([[4.17, 4.62, 3.71, 3.28]]).unsqueeze(-1).unsqueeze(-1).to(x.device)
            # x = x + torch.tensor([[5.81, 3.25, 0.12, -2.15]]).unsqueeze(-1).unsqueeze(-1).to(x.device)
            x = self.vae.decode(x).sample
            x = (x / 2 + 0.5).clamp(-1, 1)
        return x


    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.latent_vae:
            with torch.no_grad():
                img = self.vae_encode(img)

        b, c, h, w = img.shape

        # drop labels
        label = label.argmax(dim=1)
        drop = torch.rand(b, device=label.device) < 0.1
        label = torch.where(drop, self.model.n_classes, label)

        #sample time, noise, make noisy
        # each sample gets a noise between i/b and i/(b=1) to have uniform time in batch
        # time = torch.linspace(0, (b-1)/b, b) + torch.rand(b)/b
        # time = (time*self.n_timesteps).to(img.device)
        time = torch.randint(0, self.n_timesteps, (b,)).to(img.device)
        eps = torch.randn_like(img)
        img = self.scheduler.add_noise(img, eps, time)

        pred = self.model(img, time, label)
        loss = {"loss": ((pred - eps)**2).mean()}
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar = True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        img = img[0:8, ...]
        label = label[0:8, ...].argmax(dim=1)

        if self.latent_vae:
            with torch.no_grad():
                img = self.vae_encode(img)
                # img = img * 0.1

        b, c, h, w = img.shape
        #sample time, noise, make noisy
        # each sample gets a noise between i/b and i/(b=1) to have uniform time in batch
        time = torch.linspace(0, self.n_timesteps, b).to(img.device)
        # time = self.scheduler(torch.rand(b)/b + torch.arange(0, b)/b).to(img.device)
        eps = torch.randn_like(img)
        img_noisy = self.scheduler.add_noise(img, eps, time)
        pred = self.model(img_noisy, time, label)
        loss = self.loss(pred, eps, average=True)
        self.scheduler.set_timesteps(self.n_timesteps)
        _, x_0 = self.scheduler.step(pred, time, img_noisy)

        # logging
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        if self.latent_vae:
            img = self.vae_decode(img_noisy).detach()
        self.logger.log_image(
            key="image_input",
            images=[img[0], img[1], img[2], img[3],
                    img[4], img[5], img[6], img[7]]
        )
        if not self.latent_vae:
            self.logger.log_image(
                key="noise_predictions",
                images=[pred[0], pred[1], pred[2], pred[3],
                        pred[4], pred[5], pred[6], pred[7]]
            )
        if self.latent_vae:
            x_0 = self.vae_decode(x_0).detach()
        self.logger.log_image(
            key="image_predictions",
            images=[x_0[0], x_0[1], x_0[2], x_0[3],
                    x_0[4], x_0[5], x_0[6], x_0[7]]
        )

        # sample images
        label = torch.zeros_like(label)
        label[0] = 1 # goldfish
        label[1] = 9 # ostrich
        label[2] = 18 # magpie
        label[3] = 249 # malamut
        label[4] = 928 # ice cream
        label[5] = 949 # strawberry
        label[6] = 888 # viaduc
        label[7] = 409 # analog clock
        samples = torch.randn_like(img_noisy)
        samples = self.pipeline(
            samples,
            label,
            num_inference_steps=50,
            device=img.device
        )
        if self.latent_vae:
            samples = self.vae_decode(samples).detach()
        self.logger.log_image(
            key="samples",
            images=[samples[0], samples[1], samples[2], samples[3],
                    samples[4], samples[5], samples[6], samples[7]],
            caption=["goldfish", "ostrich", "magpie", "malamute",
                     "ice cream", "strawberry", "viaduc", "analog clock"]
        )


    def configure_optimizers(self):
        if self.optimizer_cfg.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.optimizer_cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,  # for lamb
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,  # for lamb
                },
            ]
            optimizer = self.optimizer_cfg.optim(optimizer_grouped_parameters)
        else:
            optimizer = self.optimizer_cfg.optim(self.model.parameters())
        scheduler = self.lr_scheduler_builder(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result