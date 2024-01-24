import argparse
import torch
from types import SimpleNamespace
from model.network.vision import HoMVision
from model.classification import ClassificationModule
from model.losses.CE import CrossEntropyLossModule
from utils.data import build_imagenet
from torch.utils.data import DataLoader
from tqdm import tqdm


device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="path to checkpoint", required=True)
parser.add_argument("--n_classes",type=int, default=1000 )
parser.add_argument("--dim", type=int, default=384)  # ok
parser.add_argument("--size", type=int, default=224)  # ok
parser.add_argument("--kernel_size", type=int, default=32)  # ok
parser.add_argument("--nb_layers", type=int, default=8)  # ok
parser.add_argument("--order", type=int, default=2)  # ok
parser.add_argument("--order_expand", type=int, default=4)  # ok
parser.add_argument("--ffw_expand", type=int, default=4)  # ok
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

# model
print('loading model from checkpoint: {}'.format(args.checkpoint))
ckpt = torch.load(args.checkpoint)
# resume_args = SimpleNamespace(**ckpt['train_config'])
# model = HoMVision(1000, resume_args.dim, resume_args.size, resume_args.kernel_size, resume_args.nb_layers,
#                   resume_args.order, resume_args.order_expand,
#                   resume_args.ffw_expand, resume_args.dropout)
model = HoMVision(1000, 384, 224, args.kernel_size, 8,
                  2, 4,
                  4, 0.)
plmodel = ClassificationModule(model, CrossEntropyLossModule, None, None, None, None, None, None)
plmodel.load_state_dict(ckpt['state_dict'])

checkpoint = {
    "model": model.state_dict(),
    "train_config": vars(args),
}
torch.save(
    checkpoint,
    "{}.ckpt".format(args.output),
)
