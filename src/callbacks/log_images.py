import logging
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch
from tqdm import tqdm

from model.diffusion import VAE

class LogGenImage(Callback):
    def __init__(self,
                log_every_n_steps: int = 5000,
                 batch_size: int = 8
                 ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.vae = VAE()
        self.ready = True
        self.last_log_step = -1
        self.batch_size = batch_size
        print(f"logging every {self.log_every_n_steps} by batches of {self.batch_size}")

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step % self.log_every_n_steps == 0 and pl_module.global_step > self.last_log_step and self.ready:
            self.last_log_step = pl_module.global_step
            x, y = batch
            print("Logging images")
            logger = trainer.logger
            # sample images
            bs = 8
            device = pl_module.device
            label = torch.zeros((bs,)).long().to(device)
            label[0] = 1 # goldfish
            label[1] = 9 # ostrich
            label[2] = 18 # magpie
            label[3] = 249 # malamut
            label[4] = 928 # ice cream
            label[5] = 949 # strawberry
            label[6] = 888 # viaduc
            label[7] = 409 # analog clock
            gen = torch.Generator(device=device)
            gen.manual_seed(3407)
            s = []
            for i in tqdm(range(8//self.batch_size)):
                samples = torch.randn(size=(self.batch_size, pl_module.model.input_dim, pl_module.model.im_size, pl_module.model.im_size),
                                      generator=gen,
                                      dtype=x.dtype,
                                      layout=x.layout,
                                      device=device)
                samples = pl_module.sampler.sample(
                    samples,
                    label[i*self.batch_size:(i+1)*self.batch_size],
                    cfg=4,
                    num_inference_steps=50,
                )
                s.append(samples)
            samples = torch.cat(s, dim=0)
            s = []
            for i in range(8):
                s.append(self.vae.vae_decode(samples[i:i+1].cpu()).detach())
            samples = torch.cat(s, dim=0)
            logger.log_image(
                key="samples",
                images=[samples[0], samples[1], samples[2], samples[3],
                        samples[4], samples[5], samples[6], samples[7]
                        ],
                caption=["goldfish", "ostrich", "magpie", "malamute",
                         "ice cream", "strawberry", "viaduc", "analog clock"
                         ]
            )
