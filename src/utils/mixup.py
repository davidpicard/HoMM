import torch
from torchvision.transforms import v2
import einops


class MixUp:
    def __init__(self, apply_transform_prob=1.0, alpha=0.1, num_classes=1000):
        super().__init__()
        self.alpha = alpha
        self._mixup = v2.MixUp(alpha=alpha, num_classes=num_classes)
        self.apply_transform_prob = apply_transform_prob

    @torch.no_grad()
    def __call__(self, x, y):
        if torch.rand(1) < self.apply_transform_prob:
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            elif y.ndim == 1:
                pass
            else:
                raise ValueError("y must be 1 or 2 dim")
            x_2 = einops.rearrange(x, "(b m) c h w -> b m c h w", m=2)
            y_2 = einops.rearrange(y, "(b m) -> b m", m=2)
            b, m = y_2.shape
            x_out = b * [None]
            y_out = b * [None]
            for i in torch.arange(0, b):
                x_, y_ = self._mixup(x_2[i], y_2[i])
                x_out[i] = x_
                y_out[i] = y_
            x_out = torch.cat(x_out, dim=0)
            y_out = torch.cat(y_out, dim=0)
            return x_out, y_out
        else:
            return x, y


class CutMix:
    def __init__(self, apply_transform_prob=1.0, alpha=1.0, num_classes=1000):
        super().__init__()
        self.alpha = alpha
        self._cutmix = v2.CutMix(alpha=alpha, num_classes=num_classes)
        self.apply_transform_prob = apply_transform_prob

    @torch.no_grad()
    def __call__(self, x, y):
        if torch.rand(1) < self.apply_transform_prob:
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            elif y.ndim == 1:
                pass
            else:
                raise ValueError("y must be 1 or 2 dim")
            x_2 = einops.rearrange(x, "(b m) c h w -> b m c h w", m=2)
            y_2 = einops.rearrange(y, "(b m) -> b m", m=2)
            b, m = y_2.shape
            x_out = b * [None]
            y_out = b * [None]
            for i in torch.arange(0, b):
                x_, y_ = self._cutmix(x_2[i], y_2[i])
                x_out[i] = x_
                y_out[i] = y_
            x_out = torch.cat(x_out, dim=0)
            y_out = torch.cat(y_out, dim=0)
            return x_out, y_out
        else:
            return x, y


class CutMixUp:
    def __init__(
        self,
        apply_transform_prob=1.0,
        mixup_prob=0.5,
        alpha_mixup=0.1,
        alpha_cutmix=1.0,
        num_classes=1000,
    ):
        self.mixup_prob = mixup_prob
        self.alpha_mixup = alpha_mixup
        self.alpha_cutmix = alpha_cutmix
        self.num_classes = num_classes
        self._mixup = v2.MixUp(alpha=alpha_mixup, num_classes=num_classes)
        self._cutmix = v2.CutMix(alpha=alpha_cutmix, num_classes=num_classes)
        self.apply_transform_prob = apply_transform_prob

    @torch.no_grad()
    def __call__(self, x, y):
        if torch.rand(1) < self.apply_transform_prob:
            if y.ndim == 2:
                y = torch.argmax(y, dim=1)
            elif y.ndim == 1:
                pass
            else:
                raise ValueError("y must be 1 or 2 dim")
            x_2 = einops.rearrange(x, "(b m) c h w -> b m c h w", m=8)
            y_2 = einops.rearrange(y, "(b m) -> b m", m=8)
            b, m = y_2.shape
            x_out = b * [None]
            y_out = b * [None]
            r = torch.rand((b,))
            for i in torch.arange(0, b):
                if r[i] < self.mixup_prob:
                    x_, y_ = self._mixup(x_2[i], y_2[i])
                else:
                    x_, y_ = self._cutmix(x_2[i], y_2[i])
                x_out[i] = x_
                y_out[i] = y_
            x_out = torch.cat(x_out, dim=0)
            y_out = torch.cat(y_out, dim=0)
            return x_out, y_out
        else:
            return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data import build_imagenet
    import sys
    import matplotlib.pyplot as plt

    data_dir = sys.argv[1]

    train, val = build_imagenet(data_dir, size=256)
    train_ds = DataLoader(train, shuffle=True, batch_size=12, num_workers=2)

    imgs, lbls = next(iter(train_ds))
    plt.subplot(4, 1, 1)
    plt.imshow(einops.rearrange(imgs, "b c h w -> h (b w) c"))

    mixup = MixUp(alpha=1.0)
    m_imgs, m_lbls = mixup(imgs, lbls)
    plt.subplot(4, 1, 2)
    plt.imshow(einops.rearrange(m_imgs, "b c h w -> h (b w) c"))

    cutmix = CutMix()
    c_imgs, m_lbls = cutmix(imgs, lbls)
    plt.subplot(4, 1, 3)
    plt.imshow(einops.rearrange(c_imgs, "b c h w -> h (b w) c"))

    cutmixup = CutMixUp()
    cm_imgs, m_lbls = cutmixup(imgs, lbls)
    plt.subplot(4, 1, 4)
    plt.imshow(einops.rearrange(cm_imgs, "b c h w -> h (b w) c"))

    plt.show()
