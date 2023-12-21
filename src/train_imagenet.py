import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchinfo
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from model.vision import HoMVision


def build_imagenet(data_dir, device="cuda", size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(1.14*size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize
    ])

    train = ImageNet(data_dir, transform=transform_train)
    val = ImageNet(data_dir, split='val', transform=transform_val)
    return train, val


def eval(model, val_ds, criterion):
    n_val = len(val_ds)
    model.eval()
    val_loss = []
    val_acc = []
    i = 1
    with tqdm(val_ds) as val:
        for imgs, lbls in val:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)
            val_loss.append(criterion(outputs, lbls).detach().cpu())
            val_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())
            val.set_postfix_str(s='val loss {:5.02f} val acc {:5.02f}'.format(torch.stack(val_loss).mean(), 100. * torch.stack(val_acc).mean()))
            i += 1
    return torch.stack(val_loss).mean().item(), 100. * torch.stack(val_acc).mean().item()


#############

dim = 128
im_size = 256
kernel_size = 16
nb_layers = 4
order = 4
order_expand = 8
ffw_expand = 4

lr = 0.00005
batch_size = 48
v_batch_size = 10
epoch = 100

device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to imagenet")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

s = args.seed
torch.manual_seed(s)

train, val = build_imagenet(args.data_dir, size=im_size)
train_ds = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
val_ds = DataLoader(val, batch_size=v_batch_size, num_workers=2)
n_train = len(train_ds)

tr_loss = []
tr_acc = []

model = HoMVision(1000, dim, im_size, kernel_size, nb_layers, order, order_expand, ffw_expand)
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler(enabled=True)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=epoch*n_train, anneal_strategy='cos', pct_start=8000/(epoch*n_train))

print('model and optimizer built')


x = torch.randn((8, 3, im_size, im_size)).to(device)
torchinfo.summary(model, input_data=x.to(device))

criterion = nn.CrossEntropyLoss()

for e in range(epoch):  # loop over the dataset multiple times
    running_loss = []
    running_acc = []
    i = 1

    with tqdm(train_ds, desc='Epoch={}'.format(e)) as tepoch:
        for imgs, lbls in tepoch:
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
            running_loss.append(loss.detach().cpu())
            running_acc.append(((outputs.argmax(dim=1) == lbls).sum() / lbls.shape[0]).detach().cpu())

            tepoch.set_postfix_str(s='loss: {:5.02f} acc: {:5.02f}'.format(torch.stack(running_loss).mean(),
                                                              100 * torch.stack(running_acc).mean(), end='\r'))

            if i % 10000 == 0:
                l, a = eval(model, val_ds, criterion)
                tr_loss.append(l)
                tr_acc.append(a)
                model.train()
            i += 1






