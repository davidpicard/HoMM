import argparse

import torch

from model.network.imagediffusion import *
from model.diffusion import DiffusionModule


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--size", required=True, type=int)
parser.add_argument("--scale", required=True, type=int)
parser.add_argument("--new-checkpoint", required=True)
args = parser.parse_args()

dihpp = False
if "pp" in args.model_name:
    model = DiHpp_models[args.model_name](n_classes=1000, input_dim=4, im_size=args.size // 8)
    dihpp=True
else:
    model = DiH_models[args.model_name](n_classes=1000, input_dim=4, im_size=args.size//8, n_timesteps=250)


class ema:
    def __init__(self):
        self.beta = 999
        self.update_after_step = 10000
        self.update_every = 10
ema_cfg = ema()

module = DiffusionModule(model, None, None, None, None, None, False, False, ema_cfg)

print('loading ckpt')
ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
module.load_state_dict(ckpt['state_dict'], strict=False)
model = module.model
print('loaded')

scale = args.scale
n_patches = model.n_patches
print('rescaling {} to {} (scale: {})'.format(model.im_size, model.im_size*scale, scale))
pos = model.pos_emb
print('expanding positional embedding (old shape: {})'.format(pos.shape))
pos = einops.rearrange(pos, "b (h w) d -> b d h w", h=n_patches)
pos_up = torch.nn.functional.interpolate(pos, scale_factor=scale, mode='bicubic', antialias=True)
pos_up = einops.rearrange(pos_up, "b d h w -> b (h w) d")
print('done (new shape: {})'.format(pos_up.shape))

ckpt['state_dict']['model.pos_emb'] = pos_up
ckpt['state_dict']['ema.ema_model.pos_emb'] = pos_up

if dihpp:
    cond_pos = model.cond_pos_emb
    print('expanding conditional pos emb (old shape: {})'.format(cond_pos.shape))
    cond_pos_r = cond_pos[:, 0:model.n_registers, :]
    print('found {} registers'.format(model.n_registers))
    pos = cond_pos[:, model.n_registers:, :]
    print('cond pos emb after removing registers: {}'.format(pos.shape))
    pos = einops.rearrange(pos, "b (h w) d -> b d h w", h=n_patches)
    pos_up = torch.nn.functional.interpolate(pos, scale_factor=scale, mode='bicubic', antialias=True)
    pos_up = einops.rearrange(pos_up, "b d h w -> b (h w) d")
    cond_pos_up = torch.cat([cond_pos_r, pos_up], dim=1)
    print('done (new shape: {})'.format(cond_pos_up.shape))

    ckpt['state_dict']['model.cond_pos_emb'] = cond_pos_up
    ckpt['state_dict']['ema.ema_model.cond_pos_emb'] = cond_pos_up

# reset step and epoch
ckpt['epoch'] = 0
ckpt['global_step'] = 0

print('saving new checkpoint')
torch.save(ckpt, args.new_checkpoint)
print('done')

