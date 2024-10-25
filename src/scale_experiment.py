import numpy as np
import torch
import time
import argparse
from tqdm import tqdm
from model.network.imagediffusion import DiH_models, DiT_models
from torch.optim import AdamW

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--loop", type=int, default=100)
parser.add_argument("--model", type=str, default="XL/2")
parser.add_argument("--backward", type=bool, default=False)
args = parser.parse_args()

dims = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256])
batch_size = args.batch_size
loop = args.loop
model_name = args.model


results = []

for d in dims:
    gen = torch.Generator()
    gen.manual_seed(3407)

    target = torch.randn((batch_size, 4, d, d), generator=gen).to(device)

    model = DiH_models[f"DiH-{model_name}"](input_dim=4, n_classes=1000, im_size=d)
    model = model.to(device)
    if not args.backward:
        model.eval()
    else:
        opt = AdamW(model.parameters(), lr=0.0001)

    # warmup
    print(f"Warmup d: {d}")
    for b in tqdm(range(10)):
        x = torch.randn((batch_size, 4, d, d), generator=gen).to(device)
        c = torch.randint(0, 1000, (batch_size,), generator=gen).to(device)
        t = torch.randint(0, 1000, (batch_size,), generator=gen).to(device)

        y_pred = model(x, c, t)

        if args.backward:
            opt.zero_grad()
            l2 = (target - y_pred).square().mean()
            l2.backward()
            opt.step()

    print(f"Test d: {d}")
    start_time = time.perf_counter()
    for b in tqdm(range(loop)):
        x = torch.randn((batch_size, 4, d, d)).to(device)
        c = torch.randint(0, 1000, (batch_size,)).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)

        y_pred = model(x, c, t)

        if args.backward:
            opt.zero_grad()
            l2 = (target - y_pred).square().mean()
            l2.backward()
            opt.step()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"d: {d}, tokens: {d*d}, elapsed: {elapsed} s, {elapsed/loop} s/batch, {elapsed/batch_size/loop} s/image")
    results.append(elapsed/batch_size/loop)

print(f"DiH images: {dims*16}")
print(f"DiH tokens: {dims**2}")
print(f"DiH s/image: {results}")

results = []

for d in dims:
    gen = torch.Generator()
    gen.manual_seed(3407)

    target = torch.randn((batch_size, 4, d, d), generator=gen).to(device)

    model = DiT_models[f"DiT-{model_name}"](input_dim=4, n_classes=1000, im_size=d)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=0.0001)

    # warmup
    print(f"Warmup d: {d}")
    for b in tqdm(range(10)):
        x = torch.randn((batch_size, 4, d, d), generator=gen).to(device)
        c = torch.randint(0, 1000, (batch_size,), generator=gen).to(device)
        t = torch.randint(0, 1000, (batch_size,), generator=gen).to(device)

        y_pred = model(x, c, t)

        if args.backward:
            opt.zero_grad()
            l2 = (target - y_pred).square().mean()
            l2.backward()
            opt.step()


    print(f"Test d: {d}")
    start_time = time.perf_counter()
    for b in tqdm(range(loop)):
        x = torch.randn((batch_size, 4, d, d)).to(device)
        c = torch.randint(0, 1000, (batch_size,)).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)

        y_pred = model(x, c, t)

        if args.backward:
            opt.zero_grad()
            l2 = (target - y_pred).square().mean()
            l2.backward()
            opt.step()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"d: {d}, tokens: {d*d}, elapsed: {elapsed} s, {elapsed/loop} s/batch, {elapsed/batch_size/loop} s/image")
    results.append(elapsed/batch_size/loop)

print(f"DiT images: {dims*16}")
print(f"DiT tokens: {dims**2}")
print(f"DiT s/image: {results}")

