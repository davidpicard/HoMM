import argparse
import os

import einops
import torch
from model.network.imagediffusion import ClassConditionalDiHpp
from model.diffusion import DiffusionModule, denormalize
from model.sampler.sampler import DiTPipeline, DDIMLinearScheduler, AncestralEulerScheduler, sigmoid_schedule, \
    linear_schedule
from torchvision.utils import save_image
from tqdm import tqdm


device = "cuda"



parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="path to checkpoint", required=True)
parser.add_argument("--size", help="image size", type=int, default=256)
parser.add_argument("--n_timesteps", type=int, default=250)
parser.add_argument("--precision", type=str, default="bf16")

# model params
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--kernel_size", type=int, default=2)
parser.add_argument("--n_layers", type=int, default=24)
parser.add_argument("--order", type=int, default=2)
parser.add_argument("--order_expand", type=int, default=2)
parser.add_argument("--ffw_expand", type=int, default=2)
parser.add_argument("--time_emb", type=int, default=1000)

parser.add_argument("--n_images_per_class", type=int, default=5)
parser.add_argument("--cfg", type=float, default=1.5)
parser.add_argument("--output", type=str, default="output")

args = parser.parse_args()

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

print("loading model")
model = ClassConditionalDiHpp(input_dim=4,
                              n_classes=1000,
                              n_timesteps=args.time_emb,
                              im_size=args.size//8,
                              kernel_size=args.kernel_size,
                              n_layers=args.n_layers,
                              dim=args.dim,
                              order=args.order,
                              order_expand=args.order_expand,
                              ffw_expand=args.ffw_expand)
class ema_cfg:
    beta = 0.999
    update_after_step = 10000
    update_every = 10

plmodule = DiffusionModule.load_from_checkpoint(args.checkpoint,
                                                model=model,
                                                mode="eps",
                                                loss=None,
                                                val_sampler=None,
                                                optimizer_cfg=None,
                                                lr_scheduler_builder=None,
                                                latent_vae=True,
                                                ema_cfg=ema_cfg())
model = plmodule.ema.ema_model.to(device)
model.eval()
vae = plmodule.vae.to(device)
vae.eval()

print("sampling images...")
with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
    pipeline = DiTPipeline(model, DDIMLinearScheduler(args.time_emb, schedule=linear_schedule))
    # pipeline = DiTPipeline(model, AncestralEulerScheduler(args.time_emb))
    for i in tqdm(range(1000)):
        torch.manual_seed(3407)
        noise = torch.randn((args.n_images_per_class, 4, args.size//8, args.size//8)).to(device)
        label = torch.zeros((args.n_images_per_class), dtype=torch.long).to(device) + i
        samples = pipeline.sample_cfg(noise, class_labels=label, cfg=args.cfg, device=device, num_inference_steps=args.n_timesteps)
        samples = vae.vae_decode(samples).detach()
        # samples = einops.rearrange(samples, "b c h w -> b h w c")
        os.makedirs("{}/{}".format(args.output, i), exist_ok=True)
        for k in range(args.n_images_per_class):
            save_image(samples[k], "{}/{}/{}.png".format(args.output, i, k))

