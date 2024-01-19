import argparse


def _add_data_params(parser):
    parser.add_argument("--data_dir", help="path to dataset", required=True)
    parser.add_argument("--data_type", default='webdataset', choices=['webdataset', 'imagefolder'])

def _add_model_params(parser):
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=16)
    parser.add_argument("--nb_layers", type=int, default=8)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--order_expand", type=int, default=8)
    parser.add_argument("--ffw_expand", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lsuv_init", type=bool, default=False)

def _add_augment_params(parser):
    parser.add_argument("--ra", type=bool, default=False)
    parser.add_argument("--ra_prob", type=float, default=0.1)
    parser.add_argument("--mixup_prob", type=float, default=1.)
    parser.add_argument("--mask_prob", type=float, default=0.5)

def _add_mae_train_params(parser):
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128, help='Cummulative batch size across all GPUs.')
    parser.add_argument("--val_batch_size", type=int, default=25)
    parser.add_argument("--max_iteration", type=int, default=300000)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--num_worker", type=int, default=32, help='Cummulative workers across all GPUs.')
    parser.add_argument("--precision", type=str, default="bf16")

def _add_lincls_train_params(parser):
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--lr', default=3., type=float)
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=0, type=float)

def _add_logger_params(parser):
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--log_freq", type=int, default=5000)
    parser.add_argument("--log_graph", type=bool, default=False)
    parser.add_argument("--version", type=int, default=0)

def _add_checkpoint_params(parser):
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--load_checkpoint", type=str, default=None)

def _add_ddp_params(parser):
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')