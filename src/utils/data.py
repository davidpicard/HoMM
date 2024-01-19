
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                 std=[1./0.229, 1./0.224, 1./0.225])

def build_imagenet(data_dir, device="cuda", size=224, additional_transforms=None):

    tr = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    if additional_transforms is not None:
        tr.extend(additional_transforms)
    transform_train = transforms.Compose(tr)

    transform_val = transforms.Compose([
        transforms.Resize(int(size/0.95)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize
    ])

    train = ImageNet(data_dir, transform=transform_train)
    val = ImageNet(data_dir, split='val', transform=transform_val)
    return train, val

def build_imagefolder(data_dir, size=224, additional_transforms=None):
    tr = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    if additional_transforms is not None:
        tr.extend(additional_transforms)
    transform_train = transforms.Compose(tr)
    return ImageFolder(data_dir, transform_train)

def build_webdataset(data_dir, size=224, additional_transforms=None):
    import webdataset as wds

    tr = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    if additional_transforms is not None:
        tr.extend(additional_transforms)
    transform_train = transforms.Compose(tr)

    dataset = (
        wds.WebDataset(data_dir, shardshuffle=True)
        .shuffle(2000)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map_tuple(transform_train, lambda json: int(json["label"]))
    )

    return dataset

def build_dataset(args, size, additional_transforms=None):
    if args.dataset_type=='webdataset':
        return build_webdataset(args.data_dir, size, additional_transforms)
    elif args.dataset_type=='imagefolder':
        return build_imagefolder(args.data_dir, size, additional_transforms)
