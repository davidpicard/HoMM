from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
import wandb

from utils.video import vae_decode_video, VideoVAE


class LogGenVideo(Callback):
    def __init__(self,
                 log_every_n_steps: int = 5000,
                 val_sample_text: str = None
                 ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.vae = VideoVAE()
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
            if pl_module.global_step % self.log_every_n_steps == 0 and pl_module.global_step > self.last_log_step and self.ready:
                self.last_log_step = pl_module.global_step
                vid, txt, mask = batch
                if self.latents is None:
                    print("no txt provided, taking from train")
                    self.latents = txt[0:2,...]
                    self.mask = mask[0:2, ...]
                    self.txt = ["_", "_"]
                print("Logging video")
                logger = trainer.logger
                # sample images
                device = pl_module.device
                gen = torch.Generator(device=device)
                gen.manual_seed(3407)
                samples = torch.randn(size=(2, vid.shape[1], vid.shape[2], vid.shape[3], vid.shape[4]),
                                      generator=gen,
                                      dtype=vid.dtype,
                                      layout=vid.layout,
                                      device=device)
                temporal_mask = pl_module.temporal_mask
                latents = torch.from_numpy(self.latents).to(device)
                mask = torch.from_numpy(self.mask).to(device)
                samples = pl_module.sampler.sample(
                    samples,
                    latents,
                    mask,
                    temporal_mask=temporal_mask,
                    cfg=6,
                    num_inference_steps=50,
                )
                vae = self.vae.to(samples.device)
                video = []
                for frames in samples:
                    v = vae_decode_video(frames.detach(), vae).cpu()
                    video.append((255*v).type(torch.uint8).permute(1,0,2,3).numpy())
                logger.log_video("video", video, fps=[16 for i in range(2)], caption=self.txt)


