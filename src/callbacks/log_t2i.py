from diffusers import AutoencoderKL
from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
import wandb



class LogT2I(Callback):
    def __init__(self,
                 log_every_n_steps: int = 5000,
                 val_sample_text: str = None
                 ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", use_safetensors=True)
        self.vae.eval()
        self.ready = True
        self.last_log_step = -1
        self.latents = None
        self.mask = None
        self.txt = None
        if val_sample_text is not None:
            x = np.load(val_sample_text)
            self.latents = x['arr_0']
            self.mask = x['arr_1']
            self.txt = x['arr_2']


    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if pl_module.global_rank == 0:
            if pl_module.global_step % self.log_every_n_steps == 0 and trainer.global_step > self.last_log_step and self.ready:
                self.last_log_step = trainer.global_step
                img_latents = batch['img_latents']
                txt_latents = batch['text_embeddings']
                mask = batch['masks']
                txt = batch['txt']
                b, n = mask.shape
                if self.latents is None:
                    print("no txt provided, taking from train")
                    self.latents = txt_latents[0:2,...].cpu().numpy()
                    self.mask = mask[0:2, ...].cpu().numpy()
                    self.txt = txt[0:2]
                print("Logging images")
                logger = trainer.logger
                # sample images
                device = pl_module.device
                gen = torch.Generator(device=device)
                images = []
                q = min(len(self.txt), 10)
                for i in range(len(self.txt)//q):
                    gen.manual_seed(3407)
                    samples = torch.randn(size=(1, img_latents.shape[1], img_latents.shape[2], img_latents.shape[3]),
                                          generator=gen,
                                          dtype=img_latents.dtype,
                                          layout=img_latents.layout,
                                          device=device).repeat(q, 1, 1, 1)
                    latents = torch.from_numpy(self.latents[q*i:q*i+q]).float().to(device)
                    mask = torch.from_numpy(self.mask[q*i:q*i+q]).float().to(device)
                    samples = pl_module.sampler.sample(
                        samples,
                        latents,
                        mask,
                        cfg=8,
                        num_inference_steps=50,
                    )
                    vae = self.vae.to(samples.device)
                    for image in samples:
                        x = vae.decode(image.unsqueeze(0)/self.vae.config.scaling_factor).sample
                        x = (x.clamp(-1, 1) / 2 + 0.5)[0].permute(1,2,0)
                        images.append(x.detach().cpu().numpy())
                logger.log_image("samples", images=images, caption=self.txt, step=trainer.global_step)


