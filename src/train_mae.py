import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchinfo
from torchvision.transforms import v2
from tqdm import tqdm

# accelerated image loading
try:
    import accimage
    from torchvision import set_image_backend
    set_image_backend('accimage')
    print('Accelerated image loading available')
except ImportError:
    print('No accelerated image loading')


from model.vision import HoMVision
from utils.data import build_imagenet, denormalize, build_imagefolder
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
            loss = criterion(outputs, nn.functional.one_hot(lbls, num_classes=1000).float()).sum(dim=1).mean().detach().cpu()
            val_loss.append(loss)
            val_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())
            val.set_postfix_str(s='val loss {:5.02f} val acc {:5.02f}'.format(torch.stack(val_loss).mean(), 100. * torch.stack(val_acc).mean()))
    return torch.stack(val_loss).mean().item(), torch.stack(val_acc).mean().item()


#############

device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to dataset", required=True)
parser.add_argument("--seed", type=int, default=3407)
# model param
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--nb_layers", type=int, default=8)
parser.add_argument("--order", type=int, default=4)
parser.add_argument("--order_expand", type=int, default=8)
parser.add_argument("--ffw_expand", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--wd", type=float, default=0.01)
# training params
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--val_batch_size", type=int, default=25)
parser.add_argument("--max_iteration", type=int, default=300000)
parser.add_argument("--warmup", type=int, default=10000)
parser.add_argument("--num_worker", type=int, default=8)
# augment param
parser.add_argument("--ra", type=bool, default=False)
parser.add_argument("--ra_prob", type=float, default=0.1)
parser.add_argument("--mixup_prob", type=float, default=1.)
parser.add_argument("--mask_prob", type=float, default=0.5)
# log param
parser.add_argument("--log_dir", type=str, default="./logs/")
parser.add_argument("--log_freq", type=int, default=5000)
parser.add_argument("--log_graph", type=bool, default=False)
# checkpoints
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()


s = args.seed
torch.manual_seed(s)

# augment
cutmix_or_mixup = CutMixUp()
randaug = [v2.RandomApply([v2.RandAugment(magnitude=6)], p=args.ra_prob)] if args.ra else None

# build dataset
train = build_imagefolder(args.data_dir, size=args.size, additional_transforms=randaug)
train_ds = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True, prefetch_factor=4, pin_memory=True, persistent_workers=True, drop_last=True)
n_train = len(train_ds)
epoch = args.max_iteration // n_train + 1

# loss crterion
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.MSELoss(reduction='none')

tr_loss = []
tr_acc = []

if args.load_checkpoint is not None:
    print('loading model from checkpoint: {}'.format(args.load_checkpoint))
    ckpt = torch.load(args.load_checkpoint)
    resume_args = SimpleNamespace(**ckpt['train_config'])
    encoder = HoMVision(args.dim, resume_args.dim, resume_args.size, resume_args.kernel_size, resume_args.nb_layers, resume_args.order, resume_args.order_expand,
                      resume_args.ffw_expand, resume_args.dropout, pooling=None)
    encoder.load_state_dict(ckpt['encoder'])
    model = encoder.to(device)
    decoder = HoMVision(3*resume_args.kernel_size**2, resume_args.dim, resume_args.size, resume_args.kernel_size, resume_args.nb_layers, resume_args.order, resume_args.order_expand,
                      resume_args.ffw_expand, resume_args.dropout, pooling=None, in_conv=False)
    decoder.load_state_dict(ckpt['decoder'])
    decoder = decoder.to(device)
    model_name = "mae_i{}_k_{}_d{}_n{}_o{}_e{}_f{}".format(resume_args.size, resume_args.kernel_size, resume_args.dim,
                                                       resume_args.nb_layers, resume_args.order, resume_args.order_expand, resume_args.ffw_expand)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler.load_state_dict(ckpt['scaler'])
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.max_iteration+1,
                                                anneal_strategy='cos', pct_start=args.warmup / args.max_iteration,
                                                last_epoch=ckpt['global_step']-1)
    start_step = ckpt['global_step']
    start_epoch = start_step//n_train
else:
    encoder = HoMVision(args.dim, args.dim, args.size, args.kernel_size, args.nb_layers, args.order, args.order_expand,
                      args.ffw_expand, args.dropout, pooling=None)
    encoder = encoder.to(device)
    decoder = HoMVision(3*args.kernel_size**2, args.dim, args.size, args.kernel_size, args.nb_layers, args.order, args.order_expand,
                      args.ffw_expand, args.dropout, pooling=None, in_conv=False)
    decoder = decoder.to(device)
    model_name = "mae_i{}_k_{}_d{}_n{}_o{}_e{}_f{}".format(args.size, args.kernel_size, args.dim,
                                                   args.nb_layers, args.order, args.order_expand, args.ffw_expand)

    optimizer = torch.optim.AdamW(params=[{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.max_iteration+1,
                                                anneal_strategy='cos', pct_start=args.warmup/args.max_iteration)
    start_step=1
    start_epoch=0

print('model and optimizer built')
print('training model {}'.format(model_name))

# loging
version = 0
path = Path(args.log_dir+"/train/"+model_name+"_{}".format(version))
if path.exists():
    while(path.exists()):
        version += 1
        path = Path(args.log_dir + "/train/" + model_name+"_{}".format(version))
train_writer = SummaryWriter(args.log_dir+"/train/"+model_name+"_{}".format(version))
# val_writer =  SummaryWriter(args.log_dir+"/val/"+model_name+"_{}".format(version))

x = torch.randn((8, 3, args.size, args.size)).to(device)
torchinfo.summary(encoder, input_data=x.to(device))
if args.log_graph:
    train_writer.add_graph(encoder, x)
train_writer.add_hparams(hparam_dict=vars(args), metric_dict={"version": version}, run_name="")

n_tokens = (args.size//args.kernel_size)**2

# big loop
i = start_step
for e in range(start_epoch, epoch):  # loop over the dataset multiple times
    with tqdm(train_ds, desc='Epoch={}'.format(e)) as tepoch:
        for imgs, lbls in tepoch:

            # to gpu
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            b, c, h, w = imgs.shape

            mask = torch.bernoulli(torch.empty((b, n_tokens)).uniform_(0, 1), p=args.mask_prob).to(device)

            masked_imgs = einops.rearrange(imgs, 'b c (m h) (n w) -> b (m n) (h w c)', c=3,
                                           m=args.size//args.kernel_size, w=args.kernel_size)
            masked_imgs = masked_imgs * mask.unsqueeze(-1)
            masked_imgs = einops.rearrange(masked_imgs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                           m=args.size//args.kernel_size, w=args.kernel_size)

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):

                imgs_enc = encoder(masked_imgs, mask) # attend only unmask tokens using mask
                outputs = decoder(imgs_enc * mask.unsqueeze(-1)) # attend all tokens but don't backprop masked ones to the encoder
                outputs = einops.rearrange(outputs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                           m=args.size//args.kernel_size, w=args.kernel_size)

                loss = criterion(outputs, imgs).mean()

            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            # print statistics
            running_loss = loss.detach().cpu()

            train_writer.add_scalar("loss", running_loss, global_step=i)
            tepoch.set_postfix_str(s='loss: {:5.02f}'.format(running_loss))

            if i % args.log_freq == 0:
                train_writer.add_images('origin', denormalize(imgs[0:16]), global_step=i)
                train_writer.add_images('masked', denormalize(masked_imgs[0:16]), global_step=i)
                train_writer.add_images('recons', denormalize(outputs[0:16].float()), global_step=i)
                checkpoint = {"encoder": encoder.state_dict(),
                              "decoder": decoder.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(),
                              "global_step": i,
                              "train_config": vars(args)
                              }
                torch.save(checkpoint, "{}/{}_{}.ckpt".format(args.checkpoint_dir, model_name, version))

            if i >= args.max_iteration:
                print('training finished, saving last model')
                checkpoint = {"encoder": encoder.state_dict(),
                              "decoder": decoder.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(),
                              "global_step": i,
                              "train_config": vars(args)
                              }
                torch.save(checkpoint, "{}/{}_{}.ckpt".format(args.checkpoint_dir, model_name, version))
                sys.exit(0)

            i += 1






