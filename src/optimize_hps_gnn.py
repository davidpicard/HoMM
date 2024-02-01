import argparse
import random
from datetime import datetime

import numpy as np
import yaml
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler
from train_gnn import gnn_train_and_eval
from utils import gnn_utils as gnn_u


def optimize_hps(
    dataset_name=None,
    device=None,
    n_gpus=None,
    hp_config_file=None,
    train_config_file=None,
    res_root_dir=None,
    n_samples=None,
):
    # Set to default values if no valid input arguments are provided
    dataset_name = "cora" if not dataset_name else dataset_name
    device = "cpu" if not device else device
    n_gpus = 0 if not n_gpus else n_gpus
    hp_config_file = (
        "./src/configs/hp_opt_gnn.yml" if not hp_config_file else hp_config_file
    )
    train_config_file = (
        "./src/configs/train_gnn.yml" if not train_config_file else train_config_file
    )
    res_root_dir = "../Results/" if not res_root_dir else res_root_dir
    n_samples = 10 if not n_samples else n_samples

    # Load default configuration parameters
    with open(hp_config_file, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Define sampling distributions
    distributions = {}
    for hp_type in ["hyperp_model", "hyperp_training"]:
        for k, v in config[hp_type].items():
            if len(v) == 1:
                distributions[k] = [v]
            elif len(v) == 2:
                if k in ["learning_rate"]:
                    distributions[k] = loguniform(*v)
                    assert all([i > 0 for i in v]) and v[0] < v[1]
                elif k in ["dropout", "weight_decay"]:
                    distributions[k] = uniform(*v)
                else:
                    distributions[k] = randint(v[0], v[1] + 1)
    sampler = ParameterSampler(
        distributions,
        n_iter=n_samples,
        random_state=int(datetime.now().timestamp()),
    )
    print(sampler.random_state)
    results = [None] * n_samples
    session_dirs = [None] * n_samples
    for ind_sample, param in enumerate(sampler):
        # Extract sample and prepare for training funtion
        print("Initialization with hyperparameters:")
        for k, v in param.items():
            if isinstance(v, list):
                v = v[0]
                param[k] = v
            if isinstance(v, np.float64):
                v = v.item()
                param[k] = v
            print(f" --{k} {v}", end="")

        # Train and eval using the current sample
        results[ind_sample], session_dirs[ind_sample] = gnn_train_and_eval(
            dataset_name=dataset_name,
            training_args={"device": device, "n_gpus": n_gpus},
            hyperp_model={
                k: v for (k, v) in param.items() if k in config["hyperp_model"]
            },
            hyperp_training={
                k: v for (k, v) in param.items() if k in config["hyperp_training"]
            },
            def_config=train_config_file,
            res_root_dir=res_root_dir,
        )
    gnn_u.create_summary_table(results, session_dirs)
    return


def main():
    parser = argparse.ArgumentParser(
        description="GAT-HoMM hyperparameter optimization. Parameters are read from a config.yml file"
    )
    parser.add_argument(
        "--dataset_name", type=str, help="The dataset name", choices=["cora"]
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The accelerator to use",
        choices=["cpu", "gpu", "tpu"],
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        help="The number of gpus to use",
    )
    parser.add_argument(
        "--hp_config_file",
        type=str,
        help="The configuration file for hyperparameter optimization",
    )
    parser.add_argument(
        "--n_samples", type=int, help="The number of considered samples"
    )
    args = parser.parse_args()
    optimize_hps(**vars(args))
    return


if __name__ == "__main__":
    main()
