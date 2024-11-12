import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model.sampler.sampler import VideoHeunVelocitySampler
from utils.video import write_video
from utils.video import vae_decode_video, VideoVAE
from model.network.videodiffusion import TVDiH_models
from model.videodiffusion import VideoDiffusionModule

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--size", required=True, type=str)
parser.add_argument("--length", required=True, type=int)
parser.add_argument("--vbench-path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--block-causal", type=bool, default=False)
parser.add_argument("--n-timesteps", type=int, default=125)
parser.add_argument("--cfg", type=float, default=2)
args = parser.parse_args()


model = TVDiH_models[args.model_name](vid_size=args.size, vid_length=args.length)

print(f"loading model {args.model_name}")
class ema_cfg:
    beta = 0.999
    update_after_step = 10000
    update_every = 10
ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
plmodule = VideoDiffusionModule(model, None, None, None, None, None, ema_cfg=ema_cfg(), block_causal=args.block_causal)
plmodule.load_state_dict(ckpt['state_dict'], strict=False)
ckpt = None
model = plmodule.ema.ema_model.to(device)
model.eval()
temporal_mask = plmodule.temporal_mask.to(device)
vid_size = args.size.split("x")
vid_size = (int(vid_size[0]), int(vid_size[1]))
plmodule = None

print("loading VAE")
vae = VideoVAE().to(device)
print("loading text encoder")
text_encoder = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.bfloat16).encoder.to(device)
text_encoder.eval()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

sampler = VideoHeunVelocitySampler(model)

gen = torch.Generator(device=device)
gen.manual_seed(3407)

txt = [
    "sunrise over the city",
]

dimension_list = [
    # 'appearance_style',
    'color',
#    'human_action',
#    'multiple_objects',
#     'object_class',
    # 'scene',
#    'spatial_relationship',
#     'subject_consistency',
    # 'temporal_flickering',
    # 'temporal_style',
    # 'overall_consistency'
]

os.makedirs("{}".format(args.output), exist_ok=True)
for dimension in dimension_list:
        # read prompt list
    with open(f'{args.vbench_path}/prompts_per_dimension/{dimension}.txt', 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]

    for prompt in prompt_list:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            # perform sampling
            print(f"processing text: {prompt}")
            tokens = tokenizer.batch_encode_plus([prompt], max_length=64,
                                                 padding="max_length", truncation=True, return_tensors="pt",
                                                 return_attention_mask=True)
            input_ids = tokens.input_ids.to(device)
            mask = (tokens.attention_mask > 0.).to(device)
            latents = text_encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state.detach()

            print(f"generating video sample")
            samples = torch.randn(size=(5, model.input_dim, args.length, vid_size[0], vid_size[1]),
                                  generator=gen,
                                  device=device)
            ones = torch.ones((5, 1, 1)).to(device)
            # print(f"temporal mask: {temporal_mask.shape}")
            # print(f"samples shape: {samples.shape}")
            with tqdm(total=args.n_timesteps) as pbar:
                samples = sampler.sample(
                    samples,
                    latents*ones,
                    mask*ones,
                    temporal_mask=temporal_mask,
                    cfg=args.cfg,
                    num_inference_steps=args.n_timesteps,
                    step_callback=lambda x, y, z: pbar.update(1),
                )

            # sample 5 videos for each prompt
            for index in range(5):
                # print(f"samples shape: {samples.shape}")
                v = vae_decode_video(samples[index].squeeze().detach(), vae, batch_size=40).cpu()
                print("writing video")
                write_video(v, f"{args.output}/{prompt}-{index}.mp4", target_fps=16)
