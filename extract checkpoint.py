#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append("src")

import torch
import einops
import matplotlib.pyplot as plt
from model.diffusion import *
from model.network.imagediffusion import *
from model.sampler.sampler import *
from tqdm.auto import tqdm

torch.set_float32_matmul_precision('medium')
plt.rcParams['figure.figsize'] = [16, 10]


# In[ ]:


im_size = 256

class ema:
    def __init__(self):
        self.beta=999
        self.update_after_step = 10000
        self.update_every = 10
        
ema_cfg = ema()


# In[ ]:


model = DiH_XL_2(n_classes=1000, input_dim=4, im_size=im_size//8)
pl_module = DiffusionModule(model, None, None, None, None, torch_compile=False, latent_encode=True, latent_decode=True, ema_cfg=ema_cfg)


# In[ ]:


ckpt = torch.load('/the/path', map_location=torch.device('cpu'))
# ckpt = torch.load('/media/opt/models/DiH_XL2_fm_cd.ckpt', map_location=torch.device('cpu'))


# In[ ]:


pl_module.load_state_dict(ckpt['state_dict'], strict=False)
model = pl_module.ema.ema_model
ckpt = None


# In[ ]:


st = model.state_dict()


# In[ ]:


filename = "/the/path/512-XL2-fm_cd1.2M.ckpt"
torch.save(st, filename)


