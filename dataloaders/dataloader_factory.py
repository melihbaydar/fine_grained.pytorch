import os
import torch
import torchvision.transforms as transforms
from dataloaders import cassava_folder


def get_dataloader(args, data_split):
    # initialize datasets and dataloaders
    # resize_res: 256 for 224, 512 for 448, 640 for 560
    resize_res = int(args.model_input_size * 1000 / 875)
    print('Transform resize resolution: ', resize_res)
    mean_vec = cassava_folder.mean_vec
    std_vec = cassava_folder.std_vec
    dataset = loader = None
    dir_path = os.path.dirname(__file__)

    if data_split == 'train':
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(args.model_input_size),
            transforms.RandomHorizontalFlip(),
            #     transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec, std=std_vec)
        ])
        dataset = cassava_folder.CassavaFolder(
            root=dir_path + '/../cassava/train', transform=train_transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True)
        print("Number of training examples: ", len(dataset))
        print("Number of classes: ", len(dataset.classes))

    elif data_split == 'test':
        test_transform = transforms.Compose([
            transforms.Resize(resize_res),
            transforms.CenterCrop(args.model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec, std=std_vec)
        ])
        dataset = cassava_folder.CassavaFolder(
            root=dir_path + '/../cassava/test', transform=test_transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=4, pin_memory=True)
        print("Number of test examples: ", len(dataset))

    elif data_split == 'extraimages':
        extra_transform = transforms.Compose([
            transforms.Resize(resize_res),
            transforms.CenterCrop(args.model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec, std=std_vec)
        ])
        dataset = cassava_folder.CassavaFolder(
            root=dir_path + '/../cassava/extraimages',
            transform=extra_transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True)
        print("Number of test examples: ", len(dataset))

    elif data_split == 'subset':
        subset_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(args.model_input_size),
            transforms.RandomHorizontalFlip(),
            #     transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec, std=std_vec)
        ])
        dataset = cassava_folder.CassavaFolder(
            root=dir_path + '/../cassava/subset',
            transform=subset_transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True)
        print("Number of test examples: ", len(dataset))

    return dataset, loader
