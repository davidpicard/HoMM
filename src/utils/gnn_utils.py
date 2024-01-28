from datetime import datetime
from pathlib import Path

import pandas as pd
import spektral
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


def spekt2torch(graph_ds, self_edges=True, device=torch.device("cpu")):
    # Extract dimensions
    gnn_n = {}
    for key, att in zip(
        ["nodes", "feats", "labels"], ["n_nodes", "n_node_features", "n_labels"]
    ):
        gnn_n[key] = getattr(graph_ds.graphs[0], att)

    # Extract and convert node features
    features = torch.tensor(graph_ds.graphs[0].x, dtype=torch.float32, device=device)

    # Extract and convert node features
    adjacency = torch.tensor(graph_ds.graphs[0].a.todense(), dtype=bool, device=device)
    if self_edges:
        adjacency[torch.eye(gnn_n["nodes"], dtype=bool)] = True

    # Extract and convert labels
    labels = torch.tensor(
        graph_ds.graphs[0].y, dtype=torch.int16, device=device
    )  # Labels
    mask = {}

    # Retrieve split masks
    for split, split_att in zip(
        ["train", "val", "test"], ["mask_tr", "mask_va", "mask_te"]
    ):
        mask[split] = torch.tensor(
            getattr(graph_ds, split_att), dtype=bool, device=device
        )
        gnn_n[split] = mask[split].sum().item()
    return features, adjacency, labels, mask, gnn_n


def get_data(name="cora"):
    ds = spektral.datasets.citation.Citation(
        name=name,
        random_split=False,
        normalize_x=False,
    )
    return spekt2torch(ds)


def get_dataloader(name="cora"):
    # Retrieve data
    features, adjacency, labels, masks, gnn_n = get_data(name=name)
    # Create Dataset
    tensor_ds = TensorDataset(
        features.unsqueeze(0),
        labels.argmax(-1).unsqueeze(0),
    )
    # Create torch.dataloader
    dataloader = DataLoader(tensor_ds, batch_size=1, shuffle=False)
    return dataloader, adjacency, masks, gnn_n


def set_missing_to_default(input_dict, default_dict):
    for k, v in default_dict.items():
        input_dict[k] = input_dict.get(k, v)
        if not input_dict[k]:
            input_dict[k] = v
    return input_dict


def parse_arguments(parser):
    # Add an argument for the path to the YAML config file
    parser.add_argument(
        "--config_path",
        default="./src/configs/train_gnn.yml",
        help="Path to the YAML config file",
    )

    # Add all the arguments with their default values
    parser.add_argument(
        "--name",
        choices=["cora"],
        help="The dataset name",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "tpu"],
        help="The device you want to run the training script on",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        help="The number of GPUs you want to use",
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        help="The number of GAT-HoMM layers",
    )
    parser.add_argument(
        "--order",
        type=int,
        help="HoMM order",
    )
    parser.add_argument(
        "--order_expand",
        type=int,
        help="HoMM order expansion",
    )
    parser.add_argument(
        "--ffw_expand",
        type=int,
        help="HoMM ffw expansion",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="HoMM dropout probability",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="ADAM's learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="ADAM's weight decay",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Max number of training epochs",
    )

    args = parser.parse_args()

    # Load default values from YAML config file
    with open(args.config_path, "r") as config_file:
        final_config = yaml.safe_load(config_file)

    # Create final configuration, inserting values passed by CLI
    for arg_name, arg_val in vars(args).items():
        if arg_val is not None:
            for subdict_key in final_config:
                if arg_name in final_config[subdict_key]:
                    final_config[subdict_key][arg_name] = arg_val

    return final_config


def create_summary_table(results, session_dirs):
    """
    Create an Excel table from the given results and session_dirs.

    Parameters:
    - results: List of dictionaries, each containing "train", "val", and "test" keys.
    - session_dirs: List of strings representing session directories.
    """

    # Extract parent folder
    parent_folder = Path(session_dirs[0]).parent

    # Create a DataFrame
    df = pd.DataFrame(results)
    df = df.map(lambda x: x.item())

    # Add the session_dir column
    df["session_dir"] = session_dirs

    # Reorder columns
    df = df[["train", "val", "test", "session_dir"]]

    # Save to Excel
    session_date = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    df.to_excel(parent_folder / f"optim_results_{session_date}.xlsx", index=False)

    return
