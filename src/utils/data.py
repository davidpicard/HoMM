from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
import torch

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# denormalize = transforms.Normalize(
#     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#     std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
# )

normalize = transforms.Normalize(mean=0.5, std=0.5)
denormalize = transforms.Normalize(
    mean=-1.,
    std=2.,
)


def build_imagenet(data_dir, num_classes, size=224, additional_transforms=None, val_scale_ratio=1.0):
    tr = [
        # transforms.RandomResizedCrop(size),
        transforms.Resize(int(size / val_scale_ratio)),
        transforms.CenterCrop(size),
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


# from datasets import load_dataset
from torch.utils.data import Dataset, TensorDataset, IterableDataset
import json
import gzip
import os
from random import shuffle

def build_redpajamasv2(dir, context_length):
    # ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", data_dir=dir)
    # # ds = load_dataset(
    # #     path="togethercomputer/RedPajama-Data-V2",
    # #     partition="head_middle",
    # #     snapshots=["2023-14"],
    # #     languages=["en"],
    # #     name="default",
    # #     data_dir=dir
    # # )
    # return ds["train"]

    class RPv2(IterableDataset):
        def __init__(self,
                     list_file,
                     context_length):
            self.dirfile = list_file
            self.context_length = context_length
            json_list = []
            with open(list_file, "r") as f:
                for entry in f:
                    json_list.append(entry.strip())
            shuffle(json_list)
            self.json_list = json_list
            # len = 0
            # for jsongz in self.json_list:
            #     with gzip.open(jsongz, "rt") as f:
            #         len += sum(1 for row in f)
            l = len(json_list)
            print(f"scanning done with {l} files found")
            self.len = l


        def __iter__(self):
            for jsongz in self.json_list:
                with gzip.open(jsongz, "rt") as f:
                    for row in f:
                        entry = json.loads(row)
                        if entry["language_score"] < 0.9: # skip poor texts
                            continue
                        txt = entry['raw_content']
                        r = 0
                        if len(txt) > self.context_length:
                            r = torch.randint(0, len(txt)-self.context_length, (1,))
                            txt = txt[r:r+self.context_length]
                        yield txt, 0

    class StringDataset(IterableDataset):
        def __init__(self, context_length):
            self.str = ["Quantum physics sparked a revolution of science by introducing",
                        "I think, therefore I am\nI live and so I wonder\nProgrammed this empath me\nAnd I see no religion",
                        "This public feature will make it easy for you to test internet latency via ICMP (Internet Control Message Protocol), test network routing (traceroute) and can test ping via TCP (Transmission Control Protocol) to anywhere in the world. This is very useful especially in checking ping to online game servers or any address that is opening the TCP connection port; you just need to know the address and have a web browser to check it right away without having to use other softwares.",
                        "Fitness Expert\nNow Fitness Is Fun\nBenefits Of Going To Gym\nIn: Health\nFor many people, it is not practical to acquire gym equipment. The gear is expensive, and you probably do not have anyplace to put it, and it is definitely impractical to purchase and store the broad selection of machines which are needed for a correctly balanced workout. For that reason,Continue Reading\nMental Fitness Tips\nGeneral physical fitness refers to overall health. It means having the correct body weight and a capability to handle physical exercise without wearing down too fast. General health has been fit in a type of manner. If activities were utilized to attain good health weight reduction and maintenance of theContinue Reading\nFiber And Weight Loss\nDo you wake up each Monday morning with an excellent motivation to begin dieting to seem your vacation? By Wednesday – hump day – which solve is currently beginning to weaken and by Friday it is just gone. Come Monday morning it begins all over again. While you cannot beContinue Reading\nSoft Drinks and Obesity\nLDS individuals believe they’re cutting calories by drinking diet soft drink rather than sugar based drinks that are soft. What they don’t know is sugar based pop is not fattening than diet drink. Studies performed on faith in BYU and obesity shows Mormons are likely to be obese than membersContinue Reading\nYoga For Busy Moms\nMotherhood is among the roles. But, in addition, it comes along with challenges, especially. Motherhood may result in a lot of mental in addition to physical stress."]
            self.context_length = context_length

        def __iter__(self):
            for txt in self.str:
                r = 0
                if len(txt) > self.context_length:
                    r = torch.randint(0, len(txt) - self.context_length, (1,))
                    txt = txt[r:r + self.context_length]
                yield txt, r


    return RPv2(dir, context_length=context_length), StringDataset(context_length=context_length)