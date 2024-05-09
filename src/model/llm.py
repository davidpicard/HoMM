import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from einops import rearrange
from torchmetrics.functional.text import perplexity

class LLMModule(L.LightningModule):
    def __init__(
            self,
            model,
            loss,
            optimizer_cfg,
            lr_scheduler_builder,
            torch_compile=False
        ):
        super().__init__()
        # do optim
        if torch_compile:
            print("compiling model")
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        self.model = model
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        # Set to False because we don't load the vae
        self.strict_loading = False

    def state_dict(self):
        # Don't save the tokenizer
        return {k: v for k, v in super().state_dict().items() if "tokenizer" not in k}

    def training_step(self, batch, batch_idx):
        txt, r = batch
        with ((torch.no_grad())):
            tokens = self.tokenizer.batch_encode_plus(txt, max_length=self.model.context_length,
                                                       padding="max_length", truncation=True, return_tensors="pt",
                                                       return_attention_mask=True)
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = 1.*(tokens.attention_mask > 0.).to(self.device)

        # create causal masks
        b, n = attention_mask.shape
        masks = torch.tril(torch.ones((self.model.context_length, self.model.context_length))).unsqueeze(0).to(self.device) # b x n x n
        masks = masks * attention_mask.unsqueeze(2)

        pred = self.model(input_ids, masks, r)[:, 0:n-1, :]
        pred = rearrange(pred, "b n d -> (b n) d")
        target = rearrange(input_ids[:, 1:], "b n -> (b n)")
        weight = rearrange(attention_mask[:, 0:n-1], "b n -> (b n)")
        xe = F.cross_entropy(pred, target, reduction='none') * weight
        target = (target * weight - 100 * (1-weight)).long()
        pred = rearrange(pred, "(b n) d -> b n d", b=b)
        target = rearrange(target, "(b n) -> b n", b=b)
        pe = perplexity(pred, target, ignore_index=-100)
        loss = {"loss": xe.mean().clamp_max(15.0), "perplexity": pe}

        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                batch_size=b,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar = True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        txt, r = batch
        if isinstance(txt, str):
            txt = [txt]
        with ((torch.no_grad())):
            tokens = self.tokenizer.batch_encode_plus(txt, max_length=self.model.context_length,
                                                       padding="max_length", truncation=True, return_tensors="pt",
                                                       return_attention_mask=True)
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = 1.*(tokens.attention_mask > 0.).to(self.device)

        # create causal masks
        b, n = attention_mask.shape
        masks = torch.tril(torch.ones((self.model.context_length, self.model.context_length))).unsqueeze(0).to(self.device) # b x n x n
        masks = masks * attention_mask.unsqueeze(2)

        pred = self.model(input_ids, masks)[:, 0:n-1, :]
        pred = rearrange(pred, "b n d -> (b n) d")
        target = rearrange(input_ids[:, 1:], "b n -> (b n)")
        weight = rearrange(attention_mask[:, 0:n-1], "b n -> (b n)")
        xe = F.cross_entropy(pred, target, reduction='none') * weight
        target = (target * weight - 100 * (1-weight)).long()
        pred = rearrange(pred, "(b n) d -> b n d", b=b)
        target = rearrange(target, "(b n) -> b n", b=b)
        pe = perplexity(pred, target, ignore_index=-100)
        loss = {"loss": xe.mean(), "perplexity": pe}

        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                batch_size=b,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar = True
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        # auto-regressive text generation
        txt = "Generative AI is going to change the world because"
        with ((torch.no_grad())):
            tokens = self.tokenizer.batch_encode_plus([txt], max_length=self.model.context_length,
                                                 padding="max_length", truncation=True, return_tensors="pt",
                                                 return_attention_mask=True)
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = 1. * (tokens.attention_mask > 0.).to(self.device)
        n = attention_mask.sum().long() - 1
        next_token = 0
        while next_token != 1 and n < 64:
            mask = torch.tril(torch.ones((self.model.context_length, self.model.context_length))).unsqueeze(0).to(
                self.device)  # b x n x n
            mask[0, :, n:] = 0  # removing extra
            mask[0, n:, :] = 0
            pred = self.model(input_ids, mask)
            # dist = torch.distributions.categorical.Categorical(logits=2 * pred[0, n - 1])
            # next_token = dist.sample().item()
            next_token = pred[0, n-1].argmax().item()
            input_ids[0, n] = next_token
            n += 1

        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # self.logger.log_text("val/text_gen", columns=["global_step", "input", "output"], data=[[self.global_step, txt, decoded]], step=self.global_step)
        self.logger.log_image(key="gen_text", images=[torch.ones((3, 2, 256)), torch.ones((3, 2, 256))], caption=[txt, decoded])


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if hasattr(self, "do_optimizer_step") and not self.do_optimizer_step:
            print("Skipping optimizer step")
            closure_result = optimizer_closure()
            if closure_result is not None:
                return closure_result
            else:
                return
        else:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def configure_optimizers(self):
        if self.optimizer_cfg.exclude_ln_and_biases_from_weight_decay:
            print("Removing LN, Embedding and biases from weight decay")
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm, nn.Embedding])
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
                    "weight_decay": self.optimizer_cfg.optim.keywords["weight_decay"],
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