import json
import os.path
import tarfile
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from tqdm import tqdm
import io
import csv

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.video import read_video, vae_encode_video, VideoVAE


class TarWriter():
    def __init__(self, dirname, chunk_size=100, split="train"):
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

    def resume(self):
        indexfile = f"{self.dir}/{self.split}index.json"
        if os.path.isfile(indexfile):
            with open(indexfile, "r") as f:
                self.filelist = json.load(f)
                print(f"resumed {len(self.filelist)} chunks from {self.dir}/{self.split}index.json")

    def check_chunk_size(self):
        if (self.current_tar is None) or (self.current_sample_count >= self.chunks_size):
            self.add_tarfile()

    def add_tarfile(self):
        self.close()
        self.current_filename = f"{self.dir}/{self.split}chunk_{len(self.filelist)}.tar"
        self.current_tar = tarfile.open(self.current_filename, "w")
        self.current_sample_count = 0
        print(f"*** open {self.current_filename}")

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.filelist.append({"filename":self.current_filename,
                                  "count":self.current_sample_count})
            with open(f"{self.dir}/{self.split}index.json", "w") as f:
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
class WebvidDataset(Dataset):
    def __init__(self, path, size, nb_frames, start, end):
        dataset_path = os.path.split(args.path)[0]
        print(f"dataset path: {dataset_path}")
        if end < 0:
            end = 999999999
        self.files = []
        self.txt = []
        count = 0
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            reader.__next__()  # skip first line
            for row in reader:
                if count >= start:
                    self.files.append(f"{dataset_path}/{row[-1]}")
                    self.txt.append(row[-2])
                count += 1
                if count > end:
                    break
        self.total_number = len(self.files)
        self.size = size
        self.nb_frames = nb_frames
        print(f"Loading {self.total_number} files at {dataset_path} from {start} to {end}")

    def __len__(self):
        return self.total_number

    def __getitem__(self, idx):
        assert idx < self.total_number
        video_path = self.files[idx]
        txt = self.txt[idx]
        try:
            video = read_video(video_path, size=self.size, start_frame=0, end_frame=self.nb_frames)
            # print(f"video shape: {video.shape}")
            if video is None:
                raise Exception
            if video.shape[1] < self.nb_frames:
                m = self.nb_frames - video.shape[1]
                video = F.pad(video, (0,0,0,0,0,m), "constant", 0)
        except:
            print(f"buggy video: {video_path}")
            video = torch.zeros((3, self.nb_frames, self.size[0], self.size[1]))
        return {"video": video, "txt": txt}




parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--size", type=str, default="80x128")
parser.add_argument("--nb-frames", type=int, default=80)
parser.add_argument("--temp-chunk-size", type=int, default=8)
parser.add_argument("--chunk-size", type=int, default=100)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)

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

vae = VideoVAE().to(args.device)
vae = torch.compile(vae)
text_encoder = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.bfloat16).encoder.to(args.device)
text_encoder.eval()
text_encoder = torch.compile(text_encoder)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

dataset_path = os.path.split(args.path)[0]
print(f"dataset path: {dataset_path}")
out = TarWriter(args.output, chunk_size=args.chunk_size, split=args.split)
out.resume()

count = 0

data = WebvidDataset(args.path, size=size, nb_frames=args.nb_frames, start=args.start, end=args.end)
data = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

for batch in tqdm(data):
    # print(f"batch: {batch}")
    videos = batch["video"]
    txts = batch["txt"]
    with torch.autocast(device_type=args.device, dtype=precision_type, enabled=True):
        videos = videos.to(args.device)
        video_latents = vae_encode_video(videos, vae, temp_chunk_size=args.temp_chunk_size)

        tokens = tokenizer.batch_encode_plus(txts, max_length=64,
                                    padding="max_length", truncation=True, return_tensors="pt",
                                    return_attention_mask=True)
        input_ids = tokens.input_ids.to(args.device)
        attention_mask = (tokens.attention_mask > 0.).to(args.device)
        text_latents = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.detach()

        for i in range(video_latents.shape[0]):
            name = f"sample_{count}.npz"
            buffer = io.BytesIO()
            np.savez(buffer, video_latents[i].float().cpu().numpy(),
                     text_latents[i].squeeze().float().cpu().numpy(),
                     attention_mask[i].squeeze().float().cpu().numpy(),
                     txts[i])
            buffer.seek(0)
            out.add_sample(name, buffer)
            count += 1
out.close()
print(f"Finished.")

