
import pytorch_lightning as L
import torch
from torchvision.transforms import transforms

denormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
)
class DiffusionModule(L.LightningModule):
    def __init__(
            self,
            model,
            mode,
            loss,
            optimizer_cfg,
            lr_scheduler_builder,
            # train_batch_preprocess,
            val_sampler,
        ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder
        # self.train_batch_preprocess = train_batch_preprocess
        self.val_sampler = val_sampler

    def training_step(self, batch, batch_idx):
        img, label = batch
        b, c, h, w = img.shape

        #sample time, noise, make noisy
        # each sample gets a noise between i/b and i/(b=1) to have uniform time in batch
        time = torch.rand(b).to(img.device)/b + torch.arange(0, b).to(img.device)/b
        eps = torch.randn_like(img)
        img = torch.sqrt(1-time).reshape(b, 1, 1, 1) * img + torch.sqrt(time).reshape(b, 1, 1, 1) * eps

        pred = self.model(img, label, time)
        loss = self.loss(pred, eps, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        b, c, h, w = img.shape
        #sample time, noise, make noisy
        time = torch.rand(b).to(img.device)
        eps = torch.randn_like(img)
        img = torch.sqrt(1-time).reshape(b, 1, 1, 1) * img + torch.sqrt(time).reshape(b, 1, 1, 1) * eps
        self.logger.log_image(
            key="image_input",
            images=[img[0], img[1], img[2], img[3],
                    img[4], img[5], img[6], img[7]]
        )
        pred = self.model(img, label, time)
        loss = self.loss(pred, eps, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        self.logger.log_image(
            key="noise_predictions",
            images=[pred[0], pred[1], pred[2], pred[3],
                    pred[4], pred[5], pred[6], pred[7]]
        )
        x_0 = (img - torch.sqrt(time).reshape(b, 1, 1, 1) * pred) / torch.sqrt(1 - time).reshape(b, 1, 1, 1)
        x_0 = torch.clamp(x_0, -1., 1)
        self.logger.log_image(
            key="image_predictions",
            images=[x_0[0], x_0[1], x_0[2], x_0[3],
                    x_0[4], x_0[5], x_0[6], x_0[7]]
        )

        # sample images
        noise = torch.randn_like(img)
        label = torch.zeros_like(label)
        label[0, 1] = 1 # goldfish
        label[1, 9] = 1 # ostrich
        label[2, 18] = 1 # magpie
        label[3, 249] = 1 # malamut
        label[4, 928] = 1 # ice cream
        label[5, 949] = 1 # strawberry
        label[6, 888] = 1 # viaduc
        label[7, 409] = 1 # analog clock
        samples = self.val_sampler.sample(noise, self.model, label)
        samples = denormalize(samples)
        self.logger.log_image(
            key="samples",
            images=[samples[0], samples[1], samples[2], samples[3],
                    samples[4], samples[5], samples[6], samples[7]],
            caption=["goldfish", "ostrich", "magpie", "malamute",
                     "ice cream", "strawberry", "viaduc", "analog clock"]
        )

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