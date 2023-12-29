import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchinfo
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageNet
from tqdm import tqdm

from model.vision import HoMVision


def build_imagenet(data_dir, device="cuda", size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        # transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(size/0.95)),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize
    ])

    train = ImageNet(data_dir, transform=transform_train)
    val = ImageNet(data_dir, split='val', transform=transform_val)
    return train, val


def eval(model, val_ds, criterion):
    model.eval()
    val_loss = []
    val_acc = []
    with tqdm(val_ds) as val:
        for imgs, lbls in val:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)
            val_loss.append(criterion(outputs, lbls).detach().cpu())
            val_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())
            val.set_postfix_str(s='val loss {:5.02f} val acc {:5.02f}'.format(torch.stack(val_loss).mean(), 100. * torch.stack(val_acc).mean()))
    return torch.stack(val_loss).mean().item(), torch.stack(val_acc).mean().item()


#############

device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to imagenet")
parser.add_argument("--seed", type=int, default=3407)
# model param
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--nb_layers", type=int, default=8)
parser.add_argument("--order", type=int, default=4)
parser.add_argument("--order_expand", type=int, default=8)
parser.add_argument("--ffw_expand", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)
# training params
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--val_batch_size", type=int, default=25)
parser.add_argument("--max_iteration", type=int, default=500000)
parser.add_argument("--num_worker", type=int, default=8)
# log param
parser.add_argument("--log_dir", type=str, default="./logs/")
parser.add_argument("--log_freq", type=int, default=5000)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
args = parser.parse_args()

s = args.seed
torch.manual_seed(s)

train, val = build_imagenet(args.data_dir, size=args.size)
train_ds = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True, persistent_workers=True)
val_ds = DataLoader(val, batch_size=args.val_batch_size, num_workers=2)
n_train = len(train_ds)
epoch = args.max_iteration // n_train + 1

cutmix = v2.CutMix(num_classes=1000)
mixup = v2.MixUp(num_classes=1000)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

tr_loss = []
tr_acc = []

model = HoMVision(1000, args.dim, args.size, args.kernel_size, args.nb_layers, args.order, args.order_expand,
                  args.ffw_expand, args.dropout)
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler(enabled=True)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.max_iteration,
                                            anneal_strategy='cos', pct_start=10000/args.max_iteration)

print('model and optimizer built')



# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.BCEWithLogitsLoss()

model_name = "i{}_k_{}_d{}_n{}_o{}_e{}_f{}".format(args.size, args.kernel_size, args.dim,
                                                   args.nb_layers, args.order, args.order_expand, args.ffw_expand)
print('training model {}'.format(model_name))
train_writer = SummaryWriter(args.log_dir+"/train/"+model_name)
val_writer =  SummaryWriter(args.log_dir+"/val/"+model_name)


x = torch.randn((8, 3, args.size, args.size)).to(device)
torchinfo.summary(model, input_data=x.to(device))
train_writer.add_graph(model, x)

i = 1
for e in range(epoch):  # loop over the dataset multiple times
    with tqdm(train_ds, desc='Epoch={}'.format(e)) as tepoch:
        for imgs, lbls in tepoch:

            # some augment
            imgs, lbls = cutmix_or_mixup(imgs, lbls)

            # to gpu
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):

                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss = loss

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
            running_acc = ((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu()

            train_writer.add_scalar("loss", running_loss, global_step=i)
            train_writer.add_scalar("acc", running_acc, global_step=i)
            tepoch.set_postfix_str(s='loss: {:5.02f} acc: {:5.02f}'.format(running_loss, 100 * running_acc))

            if i % args.log_freq == 0:
                l, a = eval(model, val_ds, criterion)
                val_writer.add_scalar("loss", l, global_step=i)
                val_writer.add_scalar("acc", a, global_step=i)
                model.train()

                checkpoint = {"model": model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(),
                              "global_step": torch.tensor(i)
                              }
                torch.save(checkpoint, "{}/{}.ckpt".format(args.checkpoint_dir, model_name))
            i += 1






