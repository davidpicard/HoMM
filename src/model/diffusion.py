
import pytorch_lightning as L
import torch


class DiffusionModule(L.LightningModule):
    def __init__(
            self,
            model,
            mode,
            loss,
            optimizer_cfg,
            lr_scheduler_builder,
            train_batch_preprocess,
            val_sampler,
        ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder
        self.train_batch_preprocess = train_batch_preprocess
        self.val_sampler = val_sampler

    def training_step(self, batch, batch_idx):
        img, label = batch
        img, label = self.train_batch_preprocess(img, label)
        b, c, h, w = img.shape

        #sample time, noise, make noisy
        time = torch.rand(b).to(img.device)
        eps = torch.randn_like(img)
        img = torch.sqrt(1-time).reshape(b, 1, 1, 1) * img + torch.sqrt(time).reshape(b, 1, 1, 1) * eps

        pred = self.model(img, label, time)
        loss = self.loss(pred, eps, average=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", 0., sync_dist=True, on_step=False, on_epoch=True)

        # noise = torch.randn_like(img)
        # samples = self.val_sampler.sample(noise, self.model, label)
        # for img, lbl in zip(samples, label):
        #     self.log_image(
        #         "val/img_{lbl}",
        #         img,
        #         sync_dist=True,
        #         on_step=True,
        #         on_epoch=True,
        #     )
    def configure_optimizers(self):
        if self.optimizer_cfg.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.optimizer_cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,  # for lamb
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,  # for lamb
                },
            ]
            optimizer = self.optimizer_cfg.optim(optimizer_grouped_parameters)
        else:
            optimizer = self.optimizer_cfg.optim(self.model.parameters())
        scheduler = self.lr_scheduler_builder(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result