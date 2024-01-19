from pathlib import Path
import torch

def lsuv_init(ds, encoder, decoder):
    from lsuv import lsuv_with_singlebatch
    for batch in train_ds:
        break
    batch = batch[0]
    encoder = lsuv_with_singlebatch(encoder, batch, device=torch.device("cpu"), verbose=False)
    decoder = lsuv_with_singlebatch(decoder, encoder(batch), device=torch.device("cpu"), verbose=False)

def get_log_path(args):
    version = args.version
    path = Path(args.log_dir + "/train/" + args.model_name + "_{}".format(version))
    while path.exists():
        version += 1
        path = Path(args.log_dir + "/train/" + args.model_name + "_{}".format(version))
    args.version = version
    return str(path)
