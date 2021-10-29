import torchvision
import torchvision.transforms as transforms
import torch

def ImageDataLoader(args):
    # Loading Dataset
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ]
    )  
    imagenet_data = torchvision.datasets.ImageFolder(
                                root=args.data_root,
                                transform=transform)

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=args.workers)  
    return data_loader
