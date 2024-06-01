import argparse
import os

import einops
import torch
from model.network.imagediffusion import ClassConditionalDiHpp, DiHpp_models
from model.diffusion import DiffusionModule, denormalize
from model.sampler.sampler import *
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
parser.add_argument("--sampler", type=str, default="ddim")
parser.add_argument("--cfg-scheduler", type=str, default="none")
parser.add_argument("--schedule", type=str, default="linear")

parser.add_argument("--model-name", type=str, default="custom")
parser.add_argument("--compile", type=bool, default=False)

args = parser.parse_args()

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

print("loading model")
if args.model_name == "DiHpp-S/2":
    print("model: DiHpp-S/2")
    model = DiHpp_models["DiHpp-S/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
elif args.model_name == "DiHpp-B/2":
    model = DiHpp_models["DiHpp-B/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
    print("model: DiHpp-B/2")
elif args.model_name == "DiHpp-L/2":
    model = DiHpp_models["DiHpp-L/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
    print("model: DiHpp-L/2")
elif args.model_name == "DiHpp-XL/2":
    model = DiHpp_models["DiHpp-XL/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
    print("model: DiHpp-XL/2")
else:
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

cfg_scheduler = None
if args.cfg_scheduler == "linear":
    print('using linear cfg scheduler')
    cfg_scheduler = linear

schedule = linear_schedule
if args.schedule == "sigmoid":
    print('using sigmoid schedule')
    schedule = sigmoid_schedule
elif args.schedule == "cosine":
    print('using cosine schedule')
    schedule = cosine_schedule

if args.compile:
    print('compiling model')
    model = torch.compile(model, fullgraph=True, dynamic=False)

print("sampling images...")
with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
    sampler = DDIMLinearScheduler(args.time_emb, schedule=schedule)
    if args.sampler == "ddpm":
        print('using DDPM')
        sampler = DDPMLinearScheduler(args.time_emb, schedule=schedule)
    pipeline = DiTPipeline(model, sampler)
    # pipeline = DiTPipeline(model, AncestralEulerScheduler(args.time_emb))
    torch.manual_seed(3407)
    for i in tqdm(range(1000)):
        noise = torch.randn((args.n_images_per_class, 4, args.size//8, args.size//8)).to(device)
        label = torch.zeros((args.n_images_per_class), dtype=torch.long).to(device) + i
        if args.n_images_per_class > 25:
            samples = []
            for b in range(args.n_images_per_class//25):
                fr = b*25
                to = min((b+1)*25, args.n_images_per_class)
                if args.cfg > 0.:
                    sample_b = pipeline.sample_cfg(noise[fr:to, ...],
                                                   class_labels=label[fr:to],
                                                   cfg=args.cfg,
                                                   device=device,
                                                   num_inference_steps=args.n_timesteps,
                                                   cfg_scheduler=cfg_scheduler)
                else:
                    sample_b = pipeline(noise[fr:to, ...], class_labels=label[fr:to], device=device, num_inference_steps=args.n_timesteps)
                samples.append(sample_b)
            samples = torch.cat(samples, dim=0)
        else:
            if args.cfg > 0.:
                samples = pipeline.sample_cfg(noise,
                                               class_labels=label,
                                               cfg=args.cfg,
                                               device=device,
                                               num_inference_steps=args.n_timesteps,
                                               cfg_scheduler=cfg_scheduler)
            else:
                samples = pipeline(noise, class_labels=label, device=device,
                                    num_inference_steps=args.n_timesteps)
        samples = vae.vae_decode(samples).detach()
        # samples = einops.rearrange(samples, "b c h w -> b h w c")
        os.makedirs("{}/{}".format(args.output, i), exist_ok=True)
        for k in range(args.n_images_per_class):
            save_image(samples[k], "{}/{}/{}.png".format(args.output, i, k))

