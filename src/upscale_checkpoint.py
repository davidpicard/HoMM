import argparse

import torch

from model.network.imagediffusion import *
from model.diffusion import DiffusionModule


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--size", required=True, type=int)
parser.add_argument("--new-size", required=True, type=int)
parser.add_argument("--new-checkpoint", required=True)
args = parser.parse_args()

if "pp" in args.model_name:
    model = DiHpp_models[args.model_name](n_classes=1000, input_dim=4, im_size=args.size // 8)
else:
    model = DiH_models[args.model_name](n_classes=1000, input_dim=4, im_size=args.size//8, n_timesteps=250)
module = DiffusionModule(model, None, None, None, None, None, False, False)

print('loading ckpt')
ckpt = torch.load(args.checkpoint)
module.load_state_dict(ckpt['state_dict'], strict=False)
model = module.model
print('loaded')

scale = args.new_size/args.size
print('rescaling {} to {} (scale: {})'.format(args.size, args.new_size, scale))
pos = model.pos_emb
print('expanding (old shape: {})'.format(pos.shape))
pos = einops.rearrange(pos, "b (h w) d -> b d h w", h=args.size//8)
pos_up = torch.nn.functional.interpolate(pos, scale_factor=scale, mode='nearest')
pos_up = einops.rearrange(pos_up, "b d h w -> b (h w) d")
print('done (new shape: {})'.format(pos_up.shape))

print('saving new checkpoint')
ckpt['state_dict']['model.pos_emb'] = pos_up
ckpt['state_dict']['ema.ema_model'] = pos_up
torch.save(ckpt, args.new_checkpoint)
print('done')

