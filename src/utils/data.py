
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageNet

def build_imagenet(data_dir, device="cuda", size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(size/0.95)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize
    ])

    train = ImageNet(data_dir, transform=transform_train)
    val = ImageNet(data_dir, split='val', transform=transform_val)
    return train, val
