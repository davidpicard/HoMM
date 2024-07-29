import einops
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from streaming import MDSWriter
from torchvision import transforms
from argparse import ArgumentParser
from torchvision.datasets import ImageNet
from tqdm import tqdm
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument("--imagenet-path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--quantization-scale", type=int, default=8)
args = parser.parse_args()


precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

## imagenet loader
tr = [
    transforms.Resize(int(args.size)),
    transforms.CenterCrop(args.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
]
transform_train = transforms.Compose(tr)

train = ImageNet(
    args.imagenet_path,
    transform=transform_train,
)
train = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

## vae
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", device="cuda:0", subfolder="vae",
                                                 use_safetensors=True)
vae = vae.to(args.device)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False


# A dictionary mapping input fields to their data types
columns = {
    'image': 'ndarray',
    'class': 'int'
}

# Shard compression, if any
compression = 'zstd'

# import matplotlib.pyplot as plt
# plt.ion()

# Save the samples as shards using MDSWriter
count = 0
with MDSWriter(out=args.output, columns=columns, compression=compression, size_limit="200MB") as out:
    for img, lbl in tqdm(train):
        count += 1
        # if count > 1000:
        #     break
        img = img.to(args.device)
        imgf = img.flip(dims=(3,))
        img = torch.cat([img, imgf], dim=0)
        lbl = torch.cat([lbl, lbl])
        with torch.autocast(device_type=args.device, dtype=precision_type, enabled=True):
            latent = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
        latentu8 = (latent * args.quantization_scale).to(torch.int8)
        # dec = vae.decode((latentu8/args.quantization_scale).float() / vae.config.scaling_factor).sample
        #
        # imgs = torch.cat([img, dec.clamp(-1., 1.)], dim=0)*0.5+0.5
        # plt.imshow(einops.rearrange(imgs.cpu(), "(k b) c h w -> (k h) (b w) c", k=2))
        # plt.show()
        # plt.pause(0.1)

        for k in range(args.batch_size*2):
            sample = {
                'image': latentu8[k, ...].squeeze().cpu().numpy(),
                'class': lbl[k].item(),
            }
            out.write(sample)