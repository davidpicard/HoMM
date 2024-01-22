import argparse

import einops
import torch
from model.network.imagediffusion import ClassConditionalHoMDiffusion
from model.diffusion import DiffusionModule, denormalize
from model.sampler.sampler import DDIM
import matplotlib.pyplot as plt

from tqdm import tqdm


device = "cuda"



parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="path to checkpoint", required=True)
parser.add_argument("--size", help="image size", type=int, default=64)
parser.add_argument("--n_timesteps", type=int, default=1000)
parser.add_argument("--precision", type=str, default="bf16")

# model params
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--kernel_size", type=int, default=8)
parser.add_argument("--n_layers", type=int, default=8)
parser.add_argument("--order", type=int, default=4)
parser.add_argument("--order_expand", type=int, default=4)
parser.add_argument("--ffw_expand", type=int, default=4)
parser.add_argument("--time_emb", type=int, default=250)

args = parser.parse_args()

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

print("loading model")
model = ClassConditionalHoMDiffusion(1000,
                                     n_timesteps=args.time_emb,
                                     im_size=args.size,
                                     kernel_size=args.kernel_size,
                                     n_layers=args.n_layers,
                                     dim=args.dim,
                                     order=args.order,
                                     order_expand=args.order_expand,
                                     ffw_expand=args.ffw_expand)
# plmodule = DiffusionModule(model=model, mode="eps", loss=None, val_sampler=None, optimizer_cfg=None, lr_scheduler_builder=None)
plmodule = DiffusionModule.load_from_checkpoint(args.checkpoint, model=model, mode="eps", loss=None, val_sampler=None, optimizer_cfg=None, lr_scheduler_builder=None)
model = model.to(device)
model.eval()

val_sampler = DDIM(n_steps=args.n_timesteps, schedule=None)

torch.manual_seed(3407)
noise = torch.randn((16, 3, args.size, args.size)).to(device)
label = torch.zeros((16, 1000), dtype=torch.long).to(device)
label[0, 1] = 1  # goldfish
label[1, 9] = 1  # ostrich
label[2, 18] = 1  # magpie
label[3, 249] = 1  # malamut
label[4, 928] = 1  # ice cream
label[5, 949] = 1  # strawberry
label[6, 888] = 1  # viaduc
label[7, 409] = 1  # analog clock
label[8, 1] = 1 # goldfish
label[9, 9] = 1 # ostrich
label[10, 18] = 1 # magpie
label[11, 249] = 1 # malamut
label[12, 928] = 1 # ice cream
label[13, 949] = 1 # strawberry
label[14, 888] = 1 # viaduc
label[15, 409] = 1 # analog clock

print("sampling...")
with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
    samples = val_sampler.sample(noise, model, label, tqdm)
    samples = denormalize(samples).detach().cpu()
    samples = einops.rearrange(samples, "(b m) c h w -> (b h) (m w) c", b=2)
    plt.imshow(samples)
    plt.show()
