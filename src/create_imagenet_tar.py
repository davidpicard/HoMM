import json
import tarfile
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from argparse import ArgumentParser
from torchvision.datasets import ImageNet
from tqdm import tqdm
from torch.utils.data import DataLoader
import io

parser = ArgumentParser()
parser.add_argument("--imagenet-path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--quantization-scale", type=float, default=8.0)
parser.add_argument("--chunk-size", type=int, default=10000)
parser.add_argument("--target-precision", type=str, default="fp16")
args = parser.parse_args()


precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

if args.target_precision == "fp16":
    quantization_scale = 1.0
else:
    quantization_scale = args.quantization_scale

print(f"Saving data in {args.target_precision}")

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
train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

## vae
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", use_safetensors=True)
vae = vae.to(args.device)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False


class TarWriter():
    def __init__(self, dirname, chunk_size=10000, quantization_scale=8.0):
        import os, io, tarfile, json
        self.dir = dirname
        self.chunks_size = chunk_size
        self.quantization_scale = quantization_scale
        try:
            os.makedirs(dirname, exist_ok=True)
        except:
            print(f"Impossible to create {dirname}")

        self.filelist = []
        self.current_tar = None
        self.current_filename = None
        self.current_sample_count = 0

    def check_chunk_size(self):
        if (self.current_tar is None) or (self.current_sample_count >= self.chunks_size):
            self.add_tarfile()

    def add_tarfile(self):
        self.close()
        self.current_filename = f"{self.dir}/chunk_{len(self.filelist)}.tar"
        self.current_tar = tarfile.open(self.current_filename, "w")
        self.current_sample_count = 0
        print(f"*** open {self.current_filename}")

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.filelist.append({"filename":self.current_filename,
                                  "count":self.current_sample_count,
                                  "quantization_scale": self.quantization_scale})
            with open(f"{self.dir}/index.json", "w") as f:
                json.dump(self.filelist, f)
            self.current_tar = None
            self.current_filename = None
            print(f"*** closed all files")

    def add_sample(self, name, buffer):
        self.check_chunk_size()
        info = tarfile.TarInfo(name)
        info.size =  buffer.getbuffer().nbytes
        self.current_tar.addfile(info, buffer)
        self.current_sample_count += 1

out = TarWriter(args.output, chunk_size=args.chunk_size, quantization_scale = args.quantization_scale)
count = 0
max_m = 0
max_s = 0
for img, lbl in tqdm(train):
    img = img.to(args.device)
    imgf = img.flip(dims=(3,))
    img = torch.cat([img, imgf], dim=0)
    lbl = torch.cat([lbl, lbl])
    with torch.autocast(device_type=args.device, dtype=precision_type, enabled=True):
        dist = vae.encode(img).latent_dist
        latent_mean = dist.mean * vae.config.scaling_factor
        latent_std = dist.std * vae.config.scaling_factor
    if args.target_precision == "int8":
        max_m = latent_mean.abs().max() if latent_mean.abs().max() > max_m else max_m
        max_s = latent_std.abs().max() if latent_std.abs().max() > max_s else max_s
        latent_mean_q = (latent_mean * quantization_scale).clamp(-127, 127).to(torch.int8)
        latent_std_q = (latent_std * 1000 * quantization_scale).clamp(-127, 127).to(torch.int8)
    else:
        latent_mean_q = latent_mean.to(torch.float16)
        latent_std_q = (latent_std*1000).to(torch.float16)

    for i in range(len(lbl)):
        name = f"sample_{count}.npz"
        buffer = io.BytesIO()
        np.savez(buffer, latent_mean_q[i, ...].cpu().numpy(), latent_std_q[i, ...].cpu().numpy(), lbl[i].cpu().numpy())
        buffer.seek(0)
        out.add_sample(name, buffer)
        count += 1
    #     if count >= 1000:
    #         break
    # if count >= 1000:
    #     break
out.close()
print(f"Finished with max_m: {max_m} max_s: {max_s}")

