import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchinfo
from torchvision.transforms import v2
from lsuv import lsuv_with_dataloader
from tqdm import tqdm

# accelerated image loading
try:
    import accimage
    from torchvision import set_image_backend

    set_image_backend("accimage")
    print("Accelerated image loading available")
except ImportError:
    print("No accelerated image loading")


from model.vision import HoMVision
from utils.data import build_imagenet
from utils.mixup import CutMixUp


def eval(model, val_ds, criterion):
    model.eval()
    val_loss = []
    val_acc = []
    with tqdm(val_ds) as val:
        for imgs, lbls in val:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)
            loss = (
                criterion(
                    outputs, nn.functional.one_hot(lbls, num_classes=1000).float()
                )
                .sum(dim=1)
                .mean()
                .detach()
                .cpu()
            )
            val_loss.append(loss)
            val_acc.append(
                ((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu()
            )
            val.set_postfix_str(
                s="val loss {:5.02f} val acc {:5.02f}".format(
                    torch.stack(val_loss).mean(), 100.0 * torch.stack(val_acc).mean()
                )
            )
    return torch.stack(val_loss).mean().item(), torch.stack(val_acc).mean().item()


#############

device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to imagenet")  # ok
parser.add_argument("--seed", type=int, default=3407)
# model param
parser.add_argument("--dim", type=int, default=128)  # ok
parser.add_argument("--size", type=int, default=256)  # ok
parser.add_argument("--kernel_size", type=int, default=16)  # ok
parser.add_argument("--nb_layers", type=int, default=8)  # ok
parser.add_argument("--order", type=int, default=4)  # ok
parser.add_argument("--order_expand", type=int, default=8)  # ok
parser.add_argument("--ffw_expand", type=int, default=4)  # ok
parser.add_argument("--dropout", type=float, default=0.0)  # ok
parser.add_argument("--wd", type=float, default=0.01)  # ok
# training params
parser.add_argument("--lr", type=float, default=0.001)  # ok
parser.add_argument("--batch_size", type=int, default=128)  # ok
parser.add_argument("--val_batch_size", type=int, default=25)  # ok
parser.add_argument("--max_iteration", type=int, default=300000)  # ok
parser.add_argument("--warmup", type=int, default=10000)  # ok
parser.add_argument("--num_worker", type=int, default=8)  # ok
parser.add_argument("--precision", type=str, default="bf16")  # ok
# augment param
parser.add_argument("--ra", type=bool, default=False)  # ok
parser.add_argument("--ra_prob", type=float, default=0.1)  # ok
parser.add_argument("--mixup_prob", type=float, default=1.0)  # ok
# log param
parser.add_argument("--log_dir", type=str, default="./logs/")  # ok
parser.add_argument("--log_freq", type=int, default=5000)  # ok
parser.add_argument("--log_graph", type=bool, default=False)  # ok
# checkpoints
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")  # ok
parser.add_argument("--load_checkpoint", type=str, default=None)  # ok
parser.add_argument("--load_weights", type=str, default=None)  # ok
args = parser.parse_args()


s = args.seed
torch.manual_seed(s)

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

# augment
cutmix_or_mixup = CutMixUp()
randaug = (
    [v2.RandomApply([v2.RandAugment(magnitude=6)], p=args.ra_prob)] if args.ra else None
)

# build dataset
train, val = build_imagenet(
    args.data_dir, size=args.size, additional_transforms=randaug
)
train_ds = DataLoader(
    train,
    batch_size=args.batch_size,
    num_workers=args.num_worker,
    shuffle=True,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
val_ds = DataLoader(val, batch_size=args.val_batch_size, num_workers=2)
n_train = len(train_ds)
epoch = args.max_iteration // n_train + 1

# loss crterion
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.BCEWithLogitsLoss(reduction="none")

tr_loss = []
tr_acc = []

if args.load_checkpoint is not None:
    print("loading model from checkpoint: {}".format(args.load_checkpoint))
    ckpt = torch.load(args.load_checkpoint)
    resume_args = SimpleNamespace(**ckpt["train_config"])
    model = HoMVision(
        1000,
        resume_args.dim,
        resume_args.size,
        resume_args.kernel_size,
        resume_args.nb_layers,
        resume_args.order,
        resume_args.order_expand,
        resume_args.ffw_expand,
        resume_args.dropout,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model_name = "i{}_k_{}_d{}_n{}_o{}_e{}_f{}".format(
        resume_args.size,
        resume_args.kernel_size,
        resume_args.dim,
        resume_args.nb_layers,
        resume_args.order,
        resume_args.order_expand,
        resume_args.ffw_expand,
    )
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler.load_state_dict(ckpt["scaler"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.max_iteration + 1,
        anneal_strategy="cos",
        pct_start=args.warmup / args.max_iteration,
        last_epoch=ckpt["global_step"] - 1,
    )
    start_step = ckpt["global_step"]
    start_epoch = start_step // n_train
else:
    model = HoMVision(
        1000,
        args.dim,
        args.size,
        args.kernel_size,
        args.nb_layers,
        args.order,
        args.order_expand,
        args.ffw_expand,
        args.dropout,
    )
    model = model.to(device)
    model = lsuv_with_dataloader(
        model, train_ds, device=torch.device(device), verbose=False
    )
    nn.init.zeros_(model.out_proj.weight)
    nn.init.constant_(model.out_proj.bias, -6.9)

    model_name = "i{}_k_{}_d{}_n{}_o{}_e{}_f{}".format(
        args.size,
        args.kernel_size,
        args.dim,
        args.nb_layers,
        args.order,
        args.order_expand,
        args.ffw_expand,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.max_iteration + 1,
        anneal_strategy="cos",
        pct_start=args.warmup / args.max_iteration,
    )
    start_step = 1
    start_epoch = 0

if args.load_weights is not None:
    print("loading weights from chekpoint: {}".format(args.load_weights))
    ckpt = torch.load(args.load_weights)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)


print("model and optimizer built")
print("training model {}".format(model_name))

# loging
version = 0
path = Path(args.log_dir + "/train/" + model_name + "_{}".format(version))
if path.exists():
    while path.exists():
        version += 1
        path = Path(args.log_dir + "/train/" + model_name + "_{}".format(version))
train_writer = SummaryWriter(
    args.log_dir + "/train/" + model_name + "_{}".format(version)
)
val_writer = SummaryWriter(args.log_dir + "/val/" + model_name + "_{}".format(version))

x = torch.randn((8, 3, args.size, args.size)).to(device)
torchinfo.summary(model, input_data=x.to(device))
if args.log_graph:
    train_writer.add_graph(model, x)
train_writer.add_hparams(
    hparam_dict=vars(args), metric_dict={"version": version}, run_name=""
)


# big loop
i = start_step
for e in range(start_epoch, epoch):  # loop over the dataset multiple times
    with tqdm(train_ds, desc="Epoch={}".format(e)) as tepoch:
        for imgs, lbls in tepoch:
            # to gpu
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # cutmix augment
            if torch.rand(1) < args.mixup_prob:
                imgs, lbls = cutmix_or_mixup(imgs, lbls)
            else:
                lbls = nn.functional.one_hot(lbls, num_classes=1000).float()

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=precision_type, enabled=True):
                outputs = model(imgs)
                loss = criterion(outputs, lbls).sum(dim=1).mean()

            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            # print statistics
            running_loss = loss.detach().cpu()
            running_acc = (
                ((outputs.argmax(dim=1) == lbls.argmax(dim=1)).sum() / lbls.shape[0])
                .detach()
                .cpu()
            )

            if i % 10 == 0:
                train_writer.add_scalar("loss", running_loss, global_step=i)
                train_writer.add_scalar("acc", running_acc, global_step=i)
            tepoch.set_postfix_str(
                s="loss: {:5.02f} acc: {:5.02f}".format(running_loss, 100 * running_acc)
            )

            if i % args.log_freq == 0:
                l, a = eval(model, val_ds, criterion)
                val_writer.add_scalar("loss", l, global_step=i)
                val_writer.add_scalar("acc", a, global_step=i)
                model.train()

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": i,
                    "train_config": vars(args),
                }
                torch.save(
                    checkpoint,
                    "{}/{}_{}.ckpt".format(args.checkpoint_dir, model_name, version),
                )

            if i >= args.max_iteration:
                print("training finished, saving last model")
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": i,
                    "train_config": vars(args),
                }
                torch.save(
                    checkpoint,
                    "{}/{}_{}.ckpt".format(args.checkpoint_dir, model_name, version),
                )
                sys.exit(0)

            i += 1