import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .layers import HoMLayer

pl.seed_everything(0, workers=True)


class HoMLayer_gnn(HoMLayer):
    def __init__(self, adjacency, dim, order, order_expand, ffw_expand, dropout=0.0):
        super().__init__(dim, order, order_expand, ffw_expand, dropout)
        self.latent_dim = order * order_expand * dim
        self.adjacency = adjacency

    def project_xq(self, xq):
        s = F.sigmoid(self.se_proj(xq))
        return s

    def project_xc(self, xc):
        h = F.tanh(self.ho_proj(self.ho_drop(xc)))
        h = list(h.chunk(self.order, dim=-1))
        for i in range(1, self.order):
            h[i] = h[i] * h[i - 1]
        h = torch.cat(h, dim=-1)
        return h

    def forward(self, xq, xc=None):
        # n_batches, n_nodes, n_feats = xq.shape
        if xc is None:
            xc = xq  # self attention
        s = self.project_xq(xq)
        h = self.project_xc(xc)

        # Selection
        sh = s * (
            torch.matmul(self.adjacency / self.adjacency.sum(-1, keepdims=True), h)
        )

        # Aggregation
        x = xq + self.ag_proj(sh)

        # ffw
        x = x + self.ffw(x)

        return x


class GAT_HOMM_Network(torch.nn.Module):
    """Graph Attention Network with HoMM. Implementation for multiclass
    node classification.
    """

    def __init__(
        self,
        adjacency,
        n_features,
        n_classes,
        n_layers=2,
        order=4,
        order_expand=8,
        ffw_expand=4,
        dropout=0.0,
    ):
        super().__init__()
        self.adjacency = adjacency
        self.n_nodes = adjacency.shape[0]
        self.layers = torch.nn.ModuleList(
            [
                HoMLayer_gnn(
                    adjacency=adjacency,
                    dim=n_features,
                    order=order,
                    order_expand=order_expand,
                    ffw_expand=ffw_expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = torch.nn.Linear(n_features, n_classes)

    def forward(self, xq):
        """Forward computation.
                Parameters
        ---------
        xq : torch.Tensor
            shape = [n_nodes, n_features]
            Query node features.

        Returns
        --------
        logits : torch.Tensor
            The predicted node logits
            shape = [n_nodes, n_classes]
        classes : torch.Tensor
            The predicted node class
            shape = [n_nodes]
        """
        for layer in self.layers:
            xq = layer(xq)

        logits = self.linear(xq)
        classes = torch.softmax(logits, -1).argmax(-1)
        return logits, classes


class GAT_HOMM_Network_PL(pl.LightningModule):
    def __init__(
        self,
        model_args,
        training_args,
        hyperp_model,
        hyperp_training,
    ):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters("hyperp_model", "hyperp_training")
        #    *list(hyperp_model.keys()), *list(hyperp_training.keys())

        self.model = GAT_HOMM_Network(**model_args, **hyperp_model)
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.masks = training_args["masks"]

    def forward(self, batch, mode="train"):
        x, y = batch
        x, yest = self.model(x)
        # Calculate the loss only on the nodes belonging to the current split
        mask = self.masks[mode]
        loss = self.loss_module(x[:, mask, :].squeeze(), y[:, mask].squeeze())
        acc = (yest[:, mask] == y[:, mask]).sum().float() / mask.sum()
        # print(f"mode={mode}, nodes={mask.sum()}")
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.hyperp_training["learning_rate"],
            weight_decay=self.hparams.hyperp_training["weight_decay"],
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)
        print(f"v_acc={acc}")

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


def create_session_dir(results_dir):
    session_date = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    session_dir = os.path.join(results_dir, session_date)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def init_trainer(training_args, hyperp_training, session_dir):
    trainer = pl.Trainer(
        default_root_dir=session_dir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="val_acc",
                filename="{epoch}-{val_acc:.3f}",
                dirpath=os.path.join(session_dir, "checkpoints"),
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_acc",
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode="max",
            ),
        ],
        accelerator=training_args["device"],
        devices=training_args["n_gpus"] if training_args["device"] == "gpu" else "auto",
        max_epochs=hyperp_training["n_epochs"],
        enable_progress_bar=False,
    )
    trainer.logger._default_hp_metric = None
    return trainer


def evaluate_best_model(model, data_loader, best_model_path):
    # test_result = trainer.test(model, dataloaders=data_loader, verbose=False)
    batch = next(iter(data_loader))
    best_model_name = os.path.basename(best_model_path)
    print(f"=== Best model ({best_model_name}) performance ===")
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    _, test_acc = model.forward(batch, mode="test")
    return {"train": train_acc, "val": val_acc, "test": test_acc}


def print_results(results):
    for k, v in results.items():
        print(f"{k}-accuracy = {v}")


def save_summary_results(results, session_dir):
    with open(session_dir + "/summary_results.txt", "w") as file:
        for key, value in results.items():
            file.write(f"{key} = {value.item()}\n")
    return


def train_and_eval_gnn(
    model_pl,
    data_loader,
    model_args,
    training_args,
    logging_args,
    hyperp_model,
    hyperp_training,
):
    # Set seeds
    pl.seed_everything(0)
    # Create session directory
    session_dir = create_session_dir(logging_args["results_dir"])
    # Initialize trainer
    trainer = init_trainer(training_args, hyperp_training, session_dir)
    # Move to proper device
    device = "cuda" if training_args["device"] == "gpu" else "cpu"
    device = torch.device(device)
    model_args["adjacency"] = model_args["adjacency"].to(device)
    # Initialize PL model
    model = model_pl(
        model_args=model_args,
        training_args=training_args,
        hyperp_model=hyperp_model,
        hyperp_training=hyperp_training,
    )
    # Fit model
    trainer.fit(model, train_dataloaders=data_loader, val_dataloaders=data_loader)
    # Retrieve best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = model_pl.load_from_checkpoint(
        best_model_path,
        model_args=model_args,
        training_args=training_args,
    )
    # Evaluate best model on test set
    results = evaluate_best_model(model, data_loader, best_model_path)
    # Print results
    print_results(results)
    # Save results
    save_summary_results(results, session_dir)
    return model, results, session_dir
