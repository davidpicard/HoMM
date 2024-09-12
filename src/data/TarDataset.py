import torch
from torch.utils.data import Dataset
import json
import tarfile
import io
import numpy as np
from collections import OrderedDict
import threading
import os

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __del__(self):
        if len(self.cache) > 0:
            key, tar = self.cache.popitem(last=False)
            tar.close()

    def get(self, key: int) -> (tarfile.TarFile, tarfile.TarInfo):
        if key not in self.cache:
            return -1, -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: (tarfile.TarFile, tarfile.TarInfo)) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            key, tar = self.cache.popitem(last=False)
            tar.close()


def build_imagenet_tar(data_dir):
    train = TarDataset(data_dir)
    val = TarDataset(data_dir)
    return train, val


class TarDataset(Dataset):
    def __init__(self, dirname, seed=3407):
        self.dir = dirname
        self.filenames = []
        self.counts = []
        self.q = []
        with open(f"{self.dir}/index.json", "r") as f:
            j = json.load(f)
            for d in j:
                self.filenames.append(d['filename'])
                self.counts.append(d['count'])
                self.q.append(d['quantization_scale'])
        self.total_count = sum(self.counts)
        assert self.total_count > 0
        self.transform = None
        self.generator = torch.Generator().manual_seed(seed)

        self.cache = LRUCache(capacity=80)
        self.lock = threading.Lock()

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        assert idx < self.total_count
        # current_file_id = 0
        # while idx >= self.counts[current_file_id]:
        #     idx -= self.counts[current_file_id]
        #     current_file_id += 1
        current_file_id = idx % len(self.filenames)
        idx = idx // len(self.filenames)
        # print(f"pid: {os.getpid()} fid: {current_file_id} idx: {idx}")

        # cehck cache
        with self.lock:
            tar, members = self.cache.get(current_file_id)
            if tar == -1:
                tar = tarfile.open(self.filenames[current_file_id], 'r')
                members = tar.getmembers()
                self.cache.put(current_file_id, (tar, members))

        member = members[idx]
        f = tar.extractfile(member)
        data = np.load(io.BytesIO(f.read()))
        m = torch.from_numpy(data['arr_0']/self.q[current_file_id]).float()
        s = torch.from_numpy(data['arr_1']/(1000*self.q[current_file_id])).float()
        l = torch.nn.functional.one_hot(torch.from_numpy(data['arr_2']), num_classes=1000)
        return m + s * torch.randn(m.shape, generator=self.generator), l

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    import torch
    import matplotlib.pyplot as plt
    import einops
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", device="cuda:0", subfolder="vae",
                                                     use_safetensors=True)
    vae = vae.to("cuda")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False


    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    train = TarDataset(args.dir)
    train = DataLoader(train, batch_size=4, shuffle=True, num_workers=2)

    plt.ion()

    for samples, l in train:

        samples = samples.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float, enabled=True):
            dec = vae.decode(samples / vae.config.scaling_factor).sample

        imgs = dec.clamp(-1., 1.)*0.5+0.5
        plt.imshow(einops.rearrange(imgs.cpu(), "(k b) c h w -> (k h) (b w) c", k=2))
        plt.title(str(l.argmax(dim=1)))
        plt.show()
        plt.pause(0.1)