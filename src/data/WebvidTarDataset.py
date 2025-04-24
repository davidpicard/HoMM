import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import Dataset
import json
import tarfile
import io
import numpy as np
from collections import OrderedDict
import threading

from utils.video import VideoVAE, vae_decode_video

from src.utils.video import write_video


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __del__(self):
        if len(self.cache) > 0:
            key, tar = self.cache.popitem(last=False)
            tar, members = tar
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
            tar, members = tar
            tar.close()


def build_webvid_tar(data_dir, length):
    train = WebvidTarDataset(data_dir, length)
    val = WebvidTarDataset(data_dir, length, split="val")
    return train, val


class WebvidTarDataset(Dataset):
    def __init__(self, dirname, length=0, split="train", seed=3407):
        self.dir = dirname
        self.length = length
        self.split = "val_" if split == "val" else ""
        self.seq_length = []
        self.filenames = []
        self.counts = []
        # self.q = []
        if ":" in self.dir:
            dir = self.dir.split(":")
        else:
            dir = [self.dir]
        for p in dir:
            print(f"opening index {p}")
            with open(f"{p}/{self.split}index.json", "r") as f:
                j = json.load(f)
                for d in j:
                    self.filenames.append(d['filename'])
                    self.counts.append(d['count'])
                    self.seq_length.append(np.int32(d.get('length', 1)))
        self.total_count = sum(self.counts)
        assert self.total_count > 0
        self.transform = None
        self.generator = torch.Generator().manual_seed(seed)
        if self.split != "val_":
            perm = torch.randperm(len(self.filenames), generator=self.generator)
            self.filenames = [self.filenames[i] for i in perm]
            self.counts = [self.counts[i] for i in perm]
            self.seq_length = [self.seq_length[i] for i in perm]

        self.cache = LRUCache(capacity=256)
        self.lock = threading.Lock()

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        assert idx < self.total_count, f"idx {idx} exceeds total count {self.total_count}"
        current_file_id = 0
        while idx >= self.counts[current_file_id]:
            idx -= self.counts[current_file_id]
            current_file_id += 1

        # check cache
        with self.lock:
            tar, members = self.cache.get(current_file_id)
            if tar == -1:
                tar = tarfile.open(self.filenames[current_file_id], 'r')
                members = tar.getmembers()
                self.cache.put(current_file_id, (tar, members))

        member = members[idx]
        f = tar.extractfile(member)
        data = np.load(io.BytesIO(f.read()))
        video_latents = torch.from_numpy(data['arr_0']).float()
        # print(f"{self.filenames[current_file_id]} {idx} v shape1: {video_latents.shape}")
        if video_latents.shape[1] == 1:
            l = max(self.length, self.seq_length[current_file_id])
            video_latents = video_latents.repeat(1, l, 1, 1)
        if self.length > 0:
            video_latents = video_latents[:, 0:self.length, :, :]
        # print(f"{self.filenames[current_file_id]} {idx} v shape2: {video_latents.shape}")
        text_latents = torch.from_numpy(data['arr_1']).float()
        mask_latents = torch.from_numpy(data['arr_2']).float()
        txt = data['arr_3'].tobytes()
        # print(f"wvtds, txt: {type(txt)}")
        return video_latents, text_latents, mask_latents, txt

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    import torch
    import matplotlib.pyplot as plt
    import cv2
    import einops
    from tqdm import tqdm
    vae = VideoVAE()
    vae = vae.to("cuda")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False


    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--length", type=int, default=1)
    args = parser.parse_args()

    if args.repair:
        print(f"Trying to repair {args.dir} split {args.split}")
        import os, re
        filelist = []
        if args.split == "train":
            split = ""
            pattern = r"^(?!val).*\.tar"
        elif args.split == "val":
            split = "val_"
            pattern = r"val.*\.tar"
        else:
            print(f"Invalid split error: {args.split}")
            exit(-1)
        files = [f for f in os.listdir(args.dir) if re.match(pattern, f)]
        files.sort()
        for f in tqdm(files):
            with tarfile.open(f"{args.dir}/{f}", "r") as tar:
                members = tar.getmembers()
                filelist.append({"filename":f"{args.dir}/{f}",
                                      "count":len(members),
                                      "length": args.length})
        print(f"found: {filelist}")
        with open(f"{args.dir}/{split}index.json", "w") as f:
            json.dump(filelist, f)
    else:

        train = WebvidTarDataset(args.dir, split=args.split)
        train = DataLoader(train, batch_size=1, shuffle=True, num_workers=2)

        print(f"train length: {len(train)}")

        plt.ion()

        for samples in train:
            frames, text, mask, ori_txt = samples
            frames = frames.to("cuda")[0]
            print(f"sample shape: {frames.shape}")
            ori_txt = ori_txt[0].decode('utf-8')
            ori_txt = ori_txt.replace('\x00', '')
            print(f"ori_txt: {ori_txt}")

            with torch.autocast(device_type="cuda", dtype=torch.float, enabled=True):
                decoded = vae_decode_video(frames, vae)
            decoded = decoded.permute(1,0,2,3).detach().cpu()
            for img in decoded:
                plt.clf()
                plt.imshow(einops.rearrange(img, "c h w -> h w c"))
                plt.title(f"tl: {text.shape} m: {mask.sum()} txt: {ori_txt}")
                plt.show()
                plt.pause(0.02)

            decoded = decoded.permute(1,0,2,3)
            write_video(decoded, "output.mp4", 16)
            break