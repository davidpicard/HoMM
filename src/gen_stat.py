
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from scipy import linalg
from tqdm.auto import tqdm
import sys
from typing import Iterable, Optional, Tuple

from data.TarDataset import TarDataset
from model.diffusion import VAE

path = sys.argv[1]
out = sys.argv[2]

batch_size=200

train = TarDataset(path)
train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8)

vae = VAE().to("cuda")
vae.vae.compile()
inception = FeatureExtractorInceptionV3("inception_model", ["2048"]).to("cuda")
inception.eval()

spatial_features = []
def get_spatial_features(model, input, output):
    spatial_features.append(output)
inception.Mixed_6d.branch1x1.register_forward_hook(get_spatial_features)

mu = []
sig = []
mu_s = []
sig_s = []
i = 1
for b in tqdm(train):
    x, y = b
    img = vae.vae_decode(x.to("cuda"))
    img = (255*img).to(torch.uint8)
    with torch.inference_mode():
        pred, = inception(img)

    pred = pred.cpu().numpy()
    spred = spatial_features.pop().movedim(1, -1)[..., :7].cpu().numpy().reshape([pred.shape[0], -1])
    mu.append(np.mean(pred, axis=0))
    sig.append(np.cov(pred, rowvar=False))
    mu_s.append(np.mean(spred, axis=0))
    sig_s.append(np.cov(spred, rowvar=False))
    i += 1
    if i > 5000:
        break

mu = np.stack(mu, axis=0)
sig = np.stack(sig, axis=0)
mu_s = np.stack(mu_s, axis=0)
sig_s = np.stack(sig_s, axis=0)

# aggregate

mu = mu.mean(axis=0)
sig = sig.sum(axis=0) * (batch_size-1) / (i*batch_size - 1)
mu_s = mu_s.mean(axis=0)
sig_s = sig_s.sum(axis=0)* (batch_size-1) / (i*batch_size - 1)

print(f"mu: {mu.shape}")
print(f"sig: {sig.shape}")
print(f"mu_s: {mu_s.shape}")
print(f"sig_s: {sig_s.shape}")

np.savez(out, np.zeros((1000, 256, 256, 3), dtype=np.uint8), mu=mu, sigma=sig, mu_s=mu_s, sigma_s=sig_s)


