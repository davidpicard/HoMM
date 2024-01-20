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
parser.add_argument("--data_dir", help="path to imagenet", required=True)
parser.add_argument("--checkpoint", help="path to checkpoint", required=True)
parser.add_argument("--size", help="image size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--val_batch_size", type=int, default=25)
parser.add_argument("--num_worker", type=int, default=4)
parser.add_argument("--precision", type=str, default="float")
args = parser.parse_args()

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

# build dataset
train, val = build_imagenet(args.data_dir, size=args.size, num_classes=1000)
# train_ds = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True, prefetch_factor=4, pin_memory=True, persistent_workers=True, drop_last=True)
val_ds = DataLoader(val, batch_size=args.val_batch_size, num_workers=2)
# n_train = len(train_ds)

# model
print('loading model from checkpoint: {}'.format(args.checkpoint))
ckpt = torch.load(args.checkpoint)
# resume_args = SimpleNamespace(**ckpt['train_config'])
# model = HoMVision(1000, resume_args.dim, resume_args.size, resume_args.kernel_size, resume_args.nb_layers,
#                   resume_args.order, resume_args.order_expand,
#                   resume_args.ffw_expand, resume_args.dropout)
model = HoMVision(1000, 384, 224, 32, 8,
                  2, 4,
                  4, 0.)
plmodel = ClassificationModule(model, CrossEntropyLossModule, None, None, None, None, None, None)
plmodel.load_state_dict(ckpt['state_dict'])
model = plmodel.model.to(device)
# model = ClassificationModule.load_from_checkpoint(args.checkpoint).to(device)
model.eval()

val_acc=[]
with tqdm(val_ds) as val:
    for imgs, lbls in val:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
            outputs = model(imgs)
        val_acc.append(((outputs.argmax(dim=1) == lbls.argmax(dim=1)).sum() / lbls.shape[0]).detach().cpu())
        val.set_postfix_str(s='val acc {:5.02f}'.format(100. * torch.stack(val_acc).mean()))
print('final accuracy: {}'.format(100.*torch.stack(val_acc).mean().item()))

