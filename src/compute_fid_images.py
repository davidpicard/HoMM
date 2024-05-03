import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Lambda, Resize, CenterCrop

from utils.inception_metrics import MultiInceptionMetrics

from tqdm import tqdm

device = "cuda"

transforms = Compose([
    Resize(256),
    CenterCrop(256),
    ToTensor(),
    Lambda(lambda x: (255.*x).type(torch.uint8))
])


parser = argparse.ArgumentParser()
parser.add_argument("--path_real", help="path to checkpoint", required=True)
parser.add_argument("--path_gen", help="path to checkpoint", required=True)

args = parser.parse_args()

metrics = MultiInceptionMetrics(compute_conditional_metrics=True, compute_conditional_metrics_per_class=False)
metrics = metrics.to(device)

gen = ImageFolder(args.path_gen, transform=transforms)
gen = DataLoader(gen, batch_size=25, num_workers=2, shuffle=False)
print('updating generated images')
for x, y in tqdm(gen):
    x = x.to(device)
    metrics.update(x, labels=y, image_type="conditional")

rea = ImageFolder(args.path_real, transform=transforms)
rea = DataLoader(rea, batch_size=25, num_workers=2, shuffle=False)
print('updating real images')
for x, y in tqdm(rea):
    x = x.to(device)
    metrics.update(x, labels=y, image_type="real")

print('computing metrics')
print(metrics.compute())

