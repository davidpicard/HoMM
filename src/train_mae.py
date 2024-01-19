import os
import sys
from pathlib import Path
from types import SimpleNamespace

import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.data.distributed as dist
import torch.distributed as dist
from torchvision.transforms import v2

import torchinfo
from utils.opts.train_mae import parse_args
from utils.misc import lsuv_init, get_log_path
from tqdm import tqdm
from utils.logger import setup_logger

# accelerated image loading
try:
    import accimage
    from torchvision import set_image_backend
    set_image_backend('accimage')
    print('Accelerated image loading available')
except ImportError:
    print('No accelerated image loading')


from model.vision import HoMVision
from utils.data import build_dataset, denormalize
from utils.mixup import CutMixUp


#############
def main(args):
    #create dirs
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # set Precision
    precision_type = torch.float

    # DDP Initialisation
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    assert dist.is_initialized()
    device_id = dist.get_rank() % torch.cuda.device_count()

    # create logger
    logger = setup_logger(distributed_rank=dist.get_rank(), name='HoMM MAE')

    # log dist info. Only logs for rank=0
    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))

    # augmentations
    cutmix_or_mixup = CutMixUp()
    randaug = [v2.RandomApply([v2.RandAugment(magnitude=6)], p=args.ra_prob)] if args.ra else None

    # build dataset
    train = build_dataset(args, size=args.size, additional_transforms=randaug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    train_ds = DataLoader(
        train, batch_size=args.batch_size//dist.get_world_size(), num_workers=args.num_worker//dist.get_world_size(),
        prefetch_factor=4, pin_memory=True, persistent_workers=True, drop_last=True, sampler=train_sampler
    )
    epoch = args.max_iteration


    # loss criterion
    criterion = nn.MSELoss(reduction='none')

    # grad scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if args.load_checkpoint:
        logger.info('loading model from checkpoint: {}'.format(args.load_checkpoint))
        ckpt = torch.load(args.load_checkpoint)
        scaler.load_state_dict(ckpt['scaler'])
        tmp = args.load_checkpoint
        args = SimpleNamespace(**ckpt['train_config'])  #TODO: Verify if this is the requirement
        args.load_checkpoint = tmp

    #set precision after args are refreshed
    if args.precision == "bf16":
        precision_type = torch.bfloat16
    elif precision_type == "fp16":
        precision_type = torch.float16

    args.model_name = "mae_i{}_k_{}_d{}_n{}_o{}_e{}_f{}".format(args.size, args.kernel_size, args.dim,
                                                           args.nb_layers, args.order, args.order_expand,
                                                           args.ffw_expand)
    encoder = HoMVision(args.dim, args.dim, args.size, args.kernel_size, args.nb_layers,
                        args.order, args.order_expand,
                        args.ffw_expand, args.dropout, pooling=None)
    decoder = HoMVision(3 * args.kernel_size ** 2, args.dim, args.size,
                        args.kernel_size, args.nb_layers, args.order, args.order_expand,
                        args.ffw_expand, args.dropout, pooling=None, in_conv=False)
    
    if args.lsuv_init:
        lsuv_init(ds, encoder, decoder)

    encoder = torch.nn.parallel.DistributedDataParallel(encoder.to(device_id),
                                                      device_ids=[device_id],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder.to(device_id),
                                                      device_ids=[device_id],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)
    if args.load_checkpoint:
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])


    optimizer = torch.optim.AdamW(params=encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.load_checkpoint:
        optimizer.load_state_dict(ckpt['optimizer'])

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr,
                                                total_steps=args.max_iteration + 1,
                                                anneal_strategy='cos', pct_start=args.warmup / args.max_iteration,
                                                last_epoch=ckpt['global_step'] - 1 if args.load_checkpoint else -1)
    start_step = ckpt['global_step'] if args.load_checkpoint is not None else 1
    start_epoch = 0
    logger.info(f'Starting step: {start_step}')
    logger.info('model and optimizer built')
    logger.info('training model {}'.format(args.model_name))

    logger.info('model and optimizer built')
    logger.info('training model {}'.format(args.model_name))

    tr_loss = []
    tr_acc = []

    # logging
    path = get_log_path(args)
    train_writer = SummaryWriter(path)

    if dist.get_rank() == 0:
        x = torch.randn((8, 3, args.size, args.size)).to(device_id)
        torchinfo.summary(encoder, input_data=x.to(device_id))
        train_writer.add_hparams(hparam_dict=vars(args), metric_dict={"version": args.version}, run_name="")
        if args.log_graph:
            train_writer.add_graph(encoder, x)

    n_tokens = (args.size//args.kernel_size)**2

    # big loop
    i = start_step
    is_iter_max = i >= args.max_iteration

    for e in range(start_epoch, epoch):  # loop over the dataset multiple times
        with tqdm(train_ds, desc='Epoch={}'.format(e), disable=dist.get_rank() != 0) as tepoch:
            for imgs, lbls in tepoch:

                # to gpu
                imgs = imgs.to(device_id)
                lbls = lbls.to(device_id)
                b, c, h, w = imgs.shape

                mask = torch.bernoulli(torch.empty((b, n_tokens)).uniform_(0, 1), p=args.mask_prob).to(device_id)
                masked_imgs = einops.rearrange(imgs, 'b c (m h) (n w) -> b (m n) (h w c)', c=3,
                                               m=args.size // args.kernel_size, w=args.kernel_size)
                masked_imgs = masked_imgs * mask.unsqueeze(-1)
                masked_imgs = einops.rearrange(masked_imgs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                               m=args.size // args.kernel_size, w=args.kernel_size)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(dtype=precision_type, enabled=True):
                    imgs_enc = encoder(masked_imgs, mask)  # attend only unmask tokens using mask
                    outputs = decoder(imgs_enc * mask.unsqueeze(
                        -1))  # attend all tokens but don't backprop masked ones to the encoder
                    outputs = einops.rearrange(outputs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                               m=args.size // args.kernel_size, w=args.kernel_size)
                    loss = criterion(outputs, imgs).mean()

                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                sched.step()

                # print statistics
                running_loss = loss.detach().cpu()
                train_writer.add_scalar(f"loss/GPU:{dist.get_rank()}", running_loss, global_step=i)
                if dist.get_rank() == 0:
                    tepoch.set_postfix_str(s='step: {} loss: {:5.02f}'.format(i, running_loss))

                if dist.get_rank() == 0:
                    train_writer.add_scalar("loss", running_loss, global_step=i)
                    if i % args.log_freq == 0:
                        train_writer.add_images('origin', denormalize(imgs[0:16]), global_step=i)
                        train_writer.add_images('masked', denormalize(masked_imgs[0:16]), global_step=i)
                        train_writer.add_images('recons', denormalize(outputs[0:16].float()), global_step=i)
                        checkpoint = {"encoder": encoder.state_dict(),
                                      "decoder": decoder.state_dict(),
                                      "optimizer": optimizer.state_dict(),
                                      "scaler": scaler.state_dict(),
                                      "global_step": i,
                                      "train_config": vars(args)
                                      }
                        torch.save(checkpoint, "{}/{}_{}.ckpt".format(args.checkpoint_dir, args.model_name, args.version))

                i += 1
                if i >= args.max_iteration: # break the single epoch loop
                    print(i, start_step, args.max_iteration)
                    is_iter_max = True
                    break
            if is_iter_max: # break the overall training loop
                break

    if dist.get_rank() == 0:
        logger.info('Training finished, saving last model')
        checkpoint = {"encoder": encoder.state_dict(),
                      "decoder": decoder.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(),
                      "global_step": i,
                      "train_config": vars(args)
                      }
        torch.save(checkpoint, "{}/{}_{}.ckpt".format(args.checkpoint_dir, args.model_name, args.version))
    dist.destroy_process_group()

if __name__ == '__main__':
    main(parse_args())
