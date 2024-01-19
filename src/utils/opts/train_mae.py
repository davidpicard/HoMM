import argparse
from .common import _add_ddp_params
from .common import _add_data_params, _add_model_params, _add_checkpoint_params
from .common import _add_augment_params, _add_logger_params, _add_mae_train_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3407)

    _add_data_params(parser)
    _add_model_params(parser)
    _add_mae_train_params(parser)
    _add_logger_params(parser)
    _add_augment_params(parser)
    _add_checkpoint_params(parser)
    _add_ddp_params(parser)
    args = parser.parse_args()
    return args