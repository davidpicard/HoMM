import argparse

import einops
import torch
from model.network.imagediffusion import ClassConditionalDiT, ClassConditionalDiHpp
from model.diffusion import DiffusionModule, denormalize
from model.sampler.sampler import DiTPipeline, DDIMLinearScheduler, AncestralEulerScheduler, cosine_schedule, \
    linear_schedule, sigmoid_schedule, DDPMLinearScheduler
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
# model = ClassConditionalDiT(input_dim=3,
#                                      n_classes=1000,
#                                      n_timesteps=args.time_emb,
#                                      im_size=args.size,
#                                      kernel_size=args.kernel_size,
#                                      n_layers=args.n_layers,
#                                      dim=args.dim,
#                                      n_heads=6,
#                                      # order=args.order,
#                                      # order_expand=args.order_expand,
#                                      ffw_expand=args.ffw_expand)
model = ClassConditionalDiHpp(input_dim=4,
                              n_classes=1000,
                              n_timesteps=1000,
                              im_size=args.size//8,
                              dim=384,
                              n_layers=12,
                              kernel_size=2,
                              order=2,
                              order_expand=2,
                              ffw_expand=3,
                              n_registers=32,
                              )

class ema_cfg:
    beta = 0.999
    update_after_step = 10000
    update_every = 10

# plmodule = DiffusionModule(model=model, mode="eps", loss=None, val_sampler=None, optimizer_cfg=None, lr_scheduler_builder=None)
plmodule = DiffusionModule.load_from_checkpoint(args.checkpoint, model=model, mode="eps", loss=None, val_sampler=None, optimizer_cfg=None, lr_scheduler_builder=None,
                                                latent_vae=True,
                                                ema_cfg=ema_cfg())
model = plmodule.ema.ema_model.to(device)
model.eval()
vae = plmodule.vae.to(device)

torch.manual_seed(3407)
noise = torch.randn((16, 3, args.size, args.size)).to(device)
label = torch.zeros((16), dtype=torch.long).to(device)
label[0] = 1 # goldfish
label[1] = 9 # ostrich
label[2] = 18 # magpie
label[3] = 249 # malamut
label[4] = 928 # ice cream
label[5] = 949 # strawberry
label[6] = 888 # viaduc
label[7] = 409 # analog clock
label[8] = 1 # goldfish
label[9] = 9 # ostrich
label[10] = 18 # magpie
label[11] = 249 # malamut
label[12] = 928 # ice cream
label[13] = 949 # strawberry
label[14] = 888 # viaduc
label[15] = 409 # analog clock

plt.ion()

def cl(t, s, x, n):

    s = vae.vae_decode(s).detach()
    x = vae.vae_decode(x).detach()
    n = vae.vae_decode(n).detach()
    img = s.clamp(-1, 1).detach().cpu()/2 + 0.5
    img = einops.rearrange(img, "(k b) c h w -> (k h) (b w) c", k=2)
    x0 = x.clamp(-1, 1).detach().cpu()/2 + 0.5
    x0 = einops.rearrange(x0, "(k b) c h w -> (k h) (b w) c", k=2)
    n = n.clamp(-1, 1).detach().cpu()/2 + 0.5
    n = einops.rearrange(n, "(k b) c h w -> (k h) (b w) c", k=2)
    plt.figure(1)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.title("sample at time: {:.3f}".format(t))
    plt.subplot(3, 1, 2)
    plt.imshow(x0)
    plt.title('x0')
    plt.subplot(3, 1, 3)
    plt.imshow(n)
    plt.title('noise prediction')
    plt.show()
    plt.pause(0.02)


print("sampling...")

with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
    # pipeline = DiTPipeline(model, AncestralEulerScheduler(args.time_emb))
    # pipeline = DiTPipeline(model, DDIMLinearScheduler(args.time_emb))
    pipeline = DiTPipeline(model, DDPMLinearScheduler(args.time_emb))

    plt.ion()
    torch.manual_seed(3407)
    noise = torch.randn((len(label), 4, args.size//8, args.size//8)).to(device)
    samples = pipeline(noise, label, device, args.n_timesteps, step_callback=cl).detach()
    samples = vae.vae_decode(samples).detach().cpu()
    img = einops.rearrange(samples, "(k b) c h w -> (k h) (b w) c", k=2)
    plt.figure(2)
    plt.ioff()
    plt.imshow(img)
    plt.show()