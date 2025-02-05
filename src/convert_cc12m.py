import json
import os.path
import tarfile
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Normalize, ToImage, ToDtype, Lambda, Compose, CenterCrop, Resize
from tqdm import tqdm
import io
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.video import read_video, vae_encode_video, VideoVAE


class TarWriter():
    def __init__(self, dirname, chunk_size=100, split="train", tar_offset=0):
        import os, io, tarfile, json
        self.dir = dirname
        self.chunks_size = chunk_size
        self.split = "val_" if split == "val" else ""
        try:
            os.makedirs(dirname, exist_ok=True)
        except:
            print(f"Impossible to create {dirname}")

        self.filelist = []
        self.current_tar = None
        self.current_filename = None
        self.current_sample_count = 0
        self.index_num = ""
        self.tar_offset = tar_offset

    def resume(self, start):
        indexfile = f"{self.dir}/{self.split}index.json"
        if os.path.isfile(indexfile):
            index_num = 0
            while os.path.isfile(f"{self.dir}/{self.split}index{index_num}.json"):
                index_num += 1
            self.index_num = index_num
            while start > 0:
                current_filename = f"{self.dir}/{self.split}chunk_{len(self.filelist)}.tar"
                self.filelist.append({"filename": current_filename,
                                      "count": min(start, self.chunks_size)})
                start -= self.chunks_size
            print(f"resumed {len(self.filelist)} chunks from {self.dir}/{self.split}index{self.index_num}.json")

    def check_chunk_size(self):
        if (self.current_tar is None) or (self.current_sample_count >= self.chunks_size):
            self.add_tarfile()

    def add_tarfile(self):
        self.close()
        self.current_filename = f"{self.dir}/{self.split}chunk_{self.tar_offset+len(self.filelist)}.tar"
        self.current_tar = tarfile.open(self.current_filename, "w")
        self.current_sample_count = 0
        print(f"*** open {self.current_filename}")

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.filelist.append({"filename":self.current_filename,
                                  "count":self.current_sample_count})
            with open(f"{self.dir}/{self.split}index{self.index_num}.json", "w") as f:
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


from torch.utils.data.dataset import Dataset
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class CC12MDataset(Dataset):
    def __init__(self, data_dir, size, nb_frames, start=0, end=99999, cache_capacity=10):
        self.data_dir = data_dir
        self.start = start
        self.end = end
        transform = Compose([
            ToImage(),
            Resize(size=int(np.min(size) * 1.2)),
            CenterCrop(size=size),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5], std=[0.5]),
            Lambda(lambda x: torch.tile(x.unsqueeze(1), (1, nb_frames, 1, 1)))
        ])

        tar_files = []
        self.transform = transform
        for i in range(start, end):
            filename =  os.path.join(data_dir, f'{i:05d}.tar')
            if os.path.isfile(filename):
                tar_files.append(filename)
        self.tar_files = tar_files
        self.samples = self._extract_samples()
        self.tar_cache = LRUCache(capacity=cache_capacity)

    def _extract_samples(self):
        samples = []
        for tar_file in self.tar_files:
            with tarfile.open(tar_file, 'r') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.jpg'):
                        base_name = member.name.split('.')[0]
                        samples.append((tar_file, base_name))
        print(f"{len(samples)} images to process in {len(self.tar_files)} tar files")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_file, base_name = self.samples[idx]

        # Check if the tar file is already opened and cached
        tar = self.tar_cache.get(tar_file)
        if tar is None:
            tar = tarfile.open(tar_file, 'r')
            self.tar_cache.put(tar_file, tar)

        # Read image
        image_member = tar.getmember(f'{base_name}.jpg')
        image_file = tar.extractfile(image_member)
        image = Image.open(image_file)
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Read text
        text_member = tar.getmember(f'{base_name}.txt')
        text_file = tar.extractfile(text_member)
        text_data = text_file.read().decode("utf-8")
        text_file.close()

        text_data = f"image: {text_data}, 300.0fps"

        video = self.transform(image)
        image_file.close()
        # print({"video": video, "txt": text_data, "name": base_name})
        return {"video": video, "txt": text_data, "name": base_name}


parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--size", type=str, default="160x256")
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--nb-frames", type=int, default=121)
parser.add_argument("--chunk-size", type=int, default=100)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=99999)

args = parser.parse_args()

if args.split == "val":
    print("Extracting val set")

precision_type = torch.float
if args.precision == "bf16":
    precision_type = torch.bfloat16
elif precision_type == "fp16":
    precision_type = torch.float16

size = args.size.split("x")
size = (int(size[0]), int(size[1]))
print(f"video size: {size}")

fps = args.fps
print(f"fps: {fps}")

vae = VideoVAE().to(args.device)
vae = torch.compile(vae)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
text_encoder = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", quantization_config=quantization_config)
text_encoder.eval()

dataset_path = os.path.split(args.path)[0]
print(f"dataset path: {dataset_path}")
out = TarWriter(args.output, chunk_size=args.chunk_size, split=args.split, tar_offset=args.start//args.chunk_size)
out.resume(args.start)

count = 0

data = CC12MDataset(args.path, size=size, nb_frames=args.nb_frames, start=args.start, end=args.end)
data = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

for batch in tqdm(data):
    # print(f"batch: {batch}")
    videos = batch["video"]
    txts = batch["txt"]
    names = batch["name"]
    with torch.autocast(device_type=args.device, dtype=precision_type, enabled=True):
        videos = videos.to(args.device)
        video_latents = vae_encode_video(videos, vae)

    tokens = tokenizer.batch_encode_plus(txts, max_length=64,
                                padding="max_length", truncation=True, return_tensors="pt",
                                return_attention_mask=True)
    input_ids = tokens.input_ids.to(args.device)
    attention_mask = (tokens.attention_mask > 0.).to(args.device)
    text_latents = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].detach()
    # print(f"tl: {text_latents.shape}")

    for i in range(video_latents.shape[0]):
        name = f"{names[i].split(".")[0]}.npz"
        buffer = io.BytesIO()
        np.savez(buffer, video_latents[i].to(torch.float16).cpu().numpy(),
                 text_latents[i].squeeze().to(torch.float16).cpu().numpy(),
                 attention_mask[i].squeeze().to(torch.float16).cpu().numpy(),
                 txts[i])
        buffer.seek(0)
        out.add_sample(name, buffer)
        count += 1
out.close()
print(f"Finished.")

