import argparse
from lightning import Trainer
import torch
from model.network.imagediffusion import *
from model.diffusion import DiffusionModule


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--size", required=True, type=int)
parser.add_argument("--new-checkpoint", required=True)
args = parser.parse_args()

dihpp = False
if "pp" in args.model_name:
    model = DiHpp_models[args.model_name](n_classes=1000, input_dim=4, im_size=args.size // 8)
    dihpp=True
else:
    model = DiH_models[args.model_name](n_classes=1000, input_dim=4, im_size=args.size//8)


class ema:
    def __init__(self):
        self.beta = 999
        self.update_after_step = 10000
        self.update_every = 10
ema_cfg = ema()

# new module
module = DiffusionModule(model, None, None, None, None, None, False, False, ema_cfg)
trainer = Trainer()
try:
    trainer.fit(module)
except:
    pass
#save and load checkpoint
print('creating new checkpoint')
trainer.save_checkpoint(args.new_checkpoint)
init_ckpt = torch.load(args.new_checkpoint, map_location=torch.device('cpu'))



print('loading original ckpt')
ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
module.load_state_dict(ckpt['state_dict'], strict=False)
model = module.model
print('loaded')

init_ckpt['state_dict'] = ckpt['state_dict']
init_ckpt['optimizer_states'] = ckpt['optimizer_states']

print('saving new checkpoint')
torch.save(init_ckpt, args.new_checkpoint)
print('done')

