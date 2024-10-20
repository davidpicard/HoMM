import argparse
import os

import einops
import torch
from model.network.imagediffusion import DiHpp_models, DiH_models, ClassConditionalDiH
from model.diffusion import DiffusionModule, denormalize, VAE
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
parser.add_argument("--clip", type=bool, default=False)
parser.add_argument("--clip-value", type=float, default=1.0)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=1000)

parser.add_argument("--model-name", type=str, default="custom")
parser.add_argument("--compile", type=bool, default=False)
parser.add_argument("--decoder", type=str, default="vae")
parser.add_argument("--batch-size", type=int, default=25)
parser.add_argument("--format", type=str, default="png")

args = parser.parse_args()

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

print("loading model")
if args.model_name == "DiH-S/2":
    print("model: DiH-S/2")
    model = DiH_models["DiH-S/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
elif args.model_name == "DiH-B/2":
    model = DiH_models["DiH-B/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
    print("model: DiH-B/2")
elif args.model_name == "DiH-L/2":
    model = DiH_models["DiH-L/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
    print("model: DiH-L/2")
elif args.model_name == "DiH-XL/2":
    model = DiH_models["DiH-XL/2"](input_dim=4, n_classes=1000, im_size=args.size//8)
    print("model: DiH-XL/2")
elif args.model_name == "DiHpp-S/2":
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
    model = ClassConditionalDiH(input_dim=4,
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

ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
plmodule = DiffusionModule(model, None, None, None, None, None, latent_encode=False, latent_decode=True, ema_cfg=ema_cfg())
plmodule.load_state_dict(ckpt['state_dict'], strict=False)
ckpt = None
model = plmodule.ema.ema_model.to(device)
model.eval()
vae = VAE().to(device)
vae.eval()
pl_module = None

if args.decoder == "consistency":
    print("using ConsistencyDecoder from OpenAI")
    from consistencydecoder import ConsistencyDecoder
    scaling_factor = vae.vae.config.scaling_factor
    vae = None
    decoder = ConsistencyDecoder(device=device)  # Model size: 2.49 GB
    class Decoder():
        def __init__(self):
            self.decoder = decoder
            self.scaling_factor = scaling_factor
        def vae_decode(self, samples):
            dec = []
            for i in range(samples.shape[0]):
                with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
                    with torch.no_grad():
                        img = self.decoder(samples[i:i + 1, ...] / self.scaling_factor) / 2. + 0.5
                        dec.append(img)
            return torch.cat(dec)
    vae = Decoder()

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
    if args.sampler == "ddpm":
        print('using DDPM with clip: {} at {} and {} schedule'.format(args.clip, args.clip_value, args.schedule))
        sampler = DDPMLinearScheduler(model, schedule=schedule, clip_img_pred=args.clip, clip_value=args.clip_value)
    elif args.sampler == "dpm":
        print('using DPM with clip: {} at {}'.format(args.clip, args.clip_value))
        sampler = DPMScheduler(model, clip_img_pred=args.clip, clip_value=args.clip_value)
    elif args.sampler == "heun":
        print('using HeunVelocity')
        sampler = HeunVelocitySampler(model)
    else:
        print('using DDIM with clip: {} at {} and {} schedule'.format(args.clip, args.clip_value, args.schedule))
        sampler = DDIMLinearScheduler(model, schedule=schedule, clip_img_pred=args.clip, clip_value=args.clip_value)
    pipeline = sampler
    torch.manual_seed(3407)
    for i in tqdm(range(0, 1000, 1)):
        noise = torch.randn((args.n_images_per_class, 4, args.size//8, args.size//8)).to(device)
        label = torch.zeros((args.n_images_per_class), dtype=torch.long).to(device) + i

        if i < args.start:
            continue
        if i >= args.stop:
            break

        if args.n_images_per_class > args.batch_size:
            samples = []
            for b in range(args.n_images_per_class//args.batch_size):
                fr = b*args.batch_size
                to = min((b+1)*args.batch_size, args.n_images_per_class)
                sample_b = pipeline.sample(noise[fr:to, ...],
                                           class_labels=label[fr:to],
                                           cfg=args.cfg,
                                           num_inference_steps=args.n_timesteps,
                                           cfg_scheduler=cfg_scheduler)
                sample_b = vae.vae_decode(sample_b).detach()
                samples.append(sample_b)
            samples = torch.cat(samples, dim=0)
        else:
            samples = pipeline.sample(noise,
                                       class_labels=label,
                                       cfg=args.cfg,
                                       num_inference_steps=args.n_timesteps,
                                       cfg_scheduler=cfg_scheduler)
            samples = vae.vae_decode(samples).detach()

        # samples = einops.rearrange(samples, "b c h w -> b h w c")
        os.makedirs("{}/{}".format(args.output, i), exist_ok=True)
        for k in range(args.n_images_per_class):
            save_image(samples[k], "{}/{}/{}.{}".format(args.output, i, k, args.format))

