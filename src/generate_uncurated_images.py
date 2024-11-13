import torch
import einops
from model.diffusion import *
from model.network.imagediffusion import *
from model.sampler.sampler import *
from tqdm.auto import tqdm
import diffusers
import os
import argparse
from PIL import Image
from torchvision.utils import save_image


class DiHppWrapper(diffusers.models.Transformer2DModel):
    def __init__(self, model):
        super().__init__(in_channels=model.input_dim,
                         out_channels=model.input_dim,
                         sample_size=model.im_size,
                         norm_num_groups=1)
        self.model = model

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states=None,
            timestep=None,
            added_cond_kwargs=None,
            class_labels=None,
            cross_attention_kwargs=None,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict: bool = True,
    ):
        out = self.model(hidden_states, timestep, class_labels)
        if not return_dict:
            return (out,)
        return diffusers.models.transformers.transformer_2d.Transformer2DModelOutput(out)

parser = argparse.ArgumentParser()
parser.add_argument("--n-images-per-class", type=int, default=8)
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--cfg", type=float, default=10)
parser.add_argument("--n-timesteps", type=int, default=250)
parser.add_argument("--hf", type=bool, default=False)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--cutoff", type=float, default=0.75)
parser.add_argument("--wu", type=float, default=0.5)
args = parser.parse_args()

device = "cuda"
im_size = args.size

model = DiH_XL_2(n_classes=1000, input_dim=4, im_size=im_size//8)

# ckpt = torch.load('/media/opt/models/DiHpp_XL2_2.5Mcd.ckpt', map_location=torch.device('cpu'))
ckpt = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
model.compile()
# pl_module.vae = pl_module.vae.to(device)
ckpt = None

if args.hf:
    dm = DiHppWrapper(model)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", use_safetensors=True)
    pl_module = None

    scheduler = diffusers.schedulers.DDPMScheduler(
                                                  clip_sample=True,
                                                  clip_sample_range=1.5,
                                                  thresholding=False,
                                                  beta_schedule="squaredcos_cap_v2",
                                                  rescale_betas_zero_snr=True
                                                  )
    pipe = diffusers.pipelines.DiTPipeline(dm, vae, scheduler)
    pipe = pipe.to("cuda")
else:
    # pipe = DDIMLinearScheduler(model, schedule=sigmoid_schedule, clip_img_pred=True, clip_value=1.5)
    pipe = UpscaleHeunVelocitySampler(model, cutoff=args.cutoff)
    vae = VAE().to(device)
    vae.eval()

for i in tqdm([18, 19, 88, 107, 113, 207, 270, 279, 291, 360, 387, 417, 703, 928, 966, 972, 978, 980]):
# for i in tqdm([113, 207, 270, 279, 291, 360, 387, 417, 703, 928, 966, 972, 978, 980]):
    generator = torch.manual_seed(3407)
    label = torch.zeros((args.n_images_per_class,)).long() + i
    os.makedirs("{}/{}".format(args.output, i), exist_ok=True)
    n = 0
    for j in range(args.n_images_per_class):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            if args.hf:
                output = pipe(class_labels=label,
                              num_inference_steps=args.n_timesteps,
                              guidance_scale=args.cfg,
                              output_type="ndarray",
                              generator=generator)
                img = (255*output.images).astype(np.uint8)
                for m in range(8):
                    filename =  "{}/{}/{}.{}".format(args.output, i, n, 'png')
                    im = Image.fromarray(img[m])
                    im.save(filename)
                    n += 1
            else:
                noise = torch.randn((args.n_images_per_class, 4, im_size // 8, im_size // 8)).to(device)
                pbar = tqdm(total=args.n_timesteps)
                samples = pipe.sample(noise,
                                          class_labels=label,
                                          cfg=args.cfg,
                                          w_u=args.wu,
                                          m=2,
                                          num_inference_steps=args.n_timesteps,
                                          step_callback=lambda x, y, z: pbar.update(1),
                                      )
                # samples = vae.vae_decode(samples).detach()
                for k in range(args.n_images_per_class):
                    dec = vae.vae_decode(samples[k:k+1, ...].detach())
                    save_image(dec, "{}/{}/{}.{}".format(args.output, i, n, 'png'))
                    n += 1
