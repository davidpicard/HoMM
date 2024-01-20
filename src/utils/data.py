from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
)


def build_imagenet(data_dir, num_classes, size=224, additional_transforms=None, val_scale_ratio=0.8):
    tr = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if additional_transforms is not None:
        tr.extend(additional_transforms)
    transform_train = transforms.Compose(tr)

    transform_val = transforms.Compose(
        [
            transforms.Resize(int(size / val_scale_ratio)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train = ImageNet(
        data_dir,
        transform=transform_train,
        target_transform=lambda x: torch.nn.functional.one_hot(
            torch.LongTensor([x]), num_classes
        )
        .float()
        .squeeze(0),
    )
    val = ImageNet(
        data_dir,
        split="val",
        transform=transform_val,
        target_transform=lambda x: torch.nn.functional.one_hot(
            torch.LongTensor([x]), num_classes
        )
        .float()
        .squeeze(0),
    )
    return train, val


def build_imagefolder(data_dir, num_classes, size=224, additional_transforms=None):
    tr = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if additional_transforms is not None:
        tr.extend(additional_transforms)
    
    transform_train = transforms.Compose(tr)
    transform_val = transforms.Compose(
        [
            transforms.Resize(int(size / 0.95)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train = ImageFolder(
        data_dir+'/train',
        transform_train,
        target_transform=lambda x: torch.nn.functional.one_hot(
            torch.LongTensor([x]), num_classes
        )
        .float()
        .squeeze(0),
    )
    val = ImageFolder(
        data_dir+'/val',
        transform_val,
        target_transform=lambda x: torch.nn.functional.one_hot(
            torch.LongTensor([x]), num_classes
        )
        .float()
        .squeeze(0),
    )
    return train, val
