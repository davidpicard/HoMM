
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from scipy import linalg
from tqdm.auto import tqdm
import sys
from typing import Iterable, Optional, Tuple
import gc

from data.TarDataset import TarDataset
from model.diffusion import VAE

path = sys.argv[1]
out = sys.argv[2]

batch_size=100

train = TarDataset(path)
train = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

vae = VAE().to("cuda")
vae.vae.compile()
inception = FeatureExtractorInceptionV3("inception_model", ["2048"]).to("cuda")
inception.eval()

spatial_features = {}
def get_spatial_features(model, input, output):
    spatial_features['spred'] = output
inception.Mixed_6d.branch1x1.register_forward_hook(get_spatial_features)

mu = 0
sig = 0
mu_s = 0
sig_s = 0
i = 0
with torch.no_grad():
    t = tqdm(train)
    for b in t:
        x, y = b
        img = vae.vae_decode(x.to("cuda"))
        img = (255*img).to(torch.uint8)
        with torch.inference_mode():
            pred, = inception(img)

        pred = pred.cpu().numpy()
        spred = spatial_features['spred'].movedim(1, -1)[..., :7].cpu().numpy().reshape([pred.shape[0], -1])
        mu += (np.mean(pred, axis=0))
        sig += (np.cov(pred, rowvar=False))
        mu_s += (np.mean(spred, axis=0))
        sig_s += (np.cov(spred, rowvar=False))
        i += 1
        if i >= 500:
            break
        t.set_postfix_str(s=f"mu: {len(mu)} sig: {len(sig)}")

    gc.collect()
    print(f" computing stats")
    mu = mu / i
    sig = sig * (batch_size-1) / (i*batch_size - 1)
    mu_s = mu_s / i
    sig_s = sig_s * (batch_size-1) / (i*batch_size - 1)

    print(f"mu: {mu.shape}")
    print(f"sig: {sig.shape}")
    print(f"mu_s: {mu_s.shape}")
    print(f"sig_s: {sig_s.shape}")

    np.savez(out, np.zeros((1000, 256, 256, 3), dtype=np.uint8), mu=mu, sigma=sig, mu_s=mu_s, sigma_s=sig_s)


