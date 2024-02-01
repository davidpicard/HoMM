import argparse
import pprint

import yaml
from model.network import gnn_layers as gnn_l
from utils import gnn_utils as gnn_u


def gnn_train_and_eval(
    dataset_name="cora",
    training_args={},
    hyperp_model={},
    hyperp_training={},
    def_config="./src/configs/train_gnn.yml",
    res_root_dir="../Results/",
):
    # Load default configuration parameters
    with open(def_config, "r") as config_file:
        def_config = yaml.safe_load(config_file)

    # Load data
    data_loader, adjacency, masks, gnn_n = gnn_u.get_dataloader(name=dataset_name)

    # Define model arguments
    model_args = {
        "adjacency": adjacency,
        "n_features": gnn_n["feats"],
        "n_classes": gnn_n["labels"],
    }

    # Define logging arguments
    logging_args = {"results_dir": res_root_dir + dataset_name + "/GAT_HOMM/"}

    # Define training arguments
    training_args = gnn_u.set_missing_to_default(
        input_dict=training_args,
        default_dict={**def_config["training_args"], "masks": masks},
    )

    # Define model hyperparameters
    hyperp_model = gnn_u.set_missing_to_default(
        input_dict=hyperp_model, default_dict=def_config["hyperp_model"]
    )

    # Define training hyperparameters
    hyperp_training = gnn_u.set_missing_to_default(
        input_dict=hyperp_training, default_dict=def_config["hyperp_training"]
    )

    # Print parameters
    pp = pprint.PrettyPrinter(indent=4)
    print("\nFinal configuration----")
    print(f"dataset_name = {dataset_name}")
    print(f"resuls folder = {logging_args['results_dir']}")
    print("training_args=")
    pp.pprint(training_args)
    print("hyperp_model=")
    pp.pprint(hyperp_model)
    print("hyperp_training=")
    pp.pprint(hyperp_training)

    model, results, session_dir = gnn_l.train_and_eval_gnn(
        model_pl=gnn_l.GAT_HOMM_Network_PL,
        data_loader=data_loader,
        model_args=model_args,
        training_args=training_args,
        logging_args=logging_args,
        hyperp_model=hyperp_model,
        hyperp_training=hyperp_training,
    )
    return results, session_dir


def main():
    parser = argparse.ArgumentParser(
        description="Training GNN-HOMM. Parameters can be provided with a config file or directly from command line."
    )
    parsed_args = gnn_u.parse_arguments(parser)
    gnn_train_and_eval(
        dataset_name=parsed_args["dataset"]["name"],
        training_args=parsed_args["training_args"],
        hyperp_model=parsed_args["hyperp_model"],
        hyperp_training=parsed_args["hyperp_training"],
    )


# Check if the script is run from the terminal
if __name__ == "__main__":
    main()
