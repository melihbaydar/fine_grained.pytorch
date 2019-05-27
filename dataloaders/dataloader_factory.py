import os
import torch
import torchvision.transforms as transforms
from dataloaders import cassava_folder


competition_root_path = '/../cassava/'
# competition_root_path = '/../../competitions/idesigner'


def get_dataloader(args, data_split, train_percentage=0.8):
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
            root=dir_path + competition_root_path, split='train',
            split_percentage=train_percentage, transform=train_transform)
        num_train_samples = len(dataset)

        if args.use_extraimages:
            extra_dataset = cassava_folder.CassavaFolder(
                root=dir_path + competition_root_path, split='extraimages',
                transform=train_transform)
            print("Number of extra samples: ", len(extra_dataset))
            dataset = torch.utils.data.ConcatDataset[dataset, extra_dataset]
            dataset.classes = dataset[0].classes

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True)

        print("Number of training samples: ", num_train_samples)
        if args.use_extraimages:
            print("Number of combined training samples: ", len(dataset))
        print("Number of classes: ", len(dataset.classes))

    elif data_split == 'val':
        val_transform = transforms.Compose([
            transforms.Resize(resize_res),
            transforms.CenterCrop(args.model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec, std=std_vec)
        ])

        dataset = cassava_folder.CassavaFolder(
            root=dir_path + competition_root_path, split='val',
            split_percentage=train_percentage, transform=val_transform)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True)

        print("Number of validation samples: ", len(dataset))
        print("Number of classes: ", len(dataset.classes))

    elif data_split == 'test':
        test_transform = transforms.Compose([
            transforms.Resize(resize_res),
            transforms.CenterCrop(args.model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec, std=std_vec)
        ])

        dataset = cassava_folder.CassavaTestFolder(
            root=dir_path + competition_root_path + '/test', transform=test_transform)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=4, pin_memory=True)

        print("Number of test samples: ", len(dataset))

    # elif data_split == 'extraimages':
    #     extra_transform = transforms.Compose([
    #         transforms.RandomRotation(15),
    #         transforms.RandomResizedCrop(args.model_input_size),
    #         transforms.RandomHorizontalFlip(),
    #         #     transforms.RandomVerticalFlip(),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean_vec, std=std_vec)
    #     ])
    #
    #     loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=args.batch_size,
    #         shuffle=True, num_workers=4, pin_memory=True)
    #     print("Number of extra images samples: ", len(dataset))

    # elif data_split == 'subset':
    #     subset_transform = transforms.Compose([
    #         transforms.RandomRotation(15),
    #         transforms.RandomResizedCrop(args.model_input_size),
    #         transforms.RandomHorizontalFlip(),
    #         #     transforms.RandomVerticalFlip(),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean_vec, std=std_vec)
    #     ])
    #     dataset = cassava_folder.CassavaFolder(
    #         root=dir_path + '/../cassava/subset',
    #         transform=subset_transform)
    #     loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=args.batch_size,
    #         shuffle=True, num_workers=4, pin_memory=True)
    #     print("Number of test examples: ", len(dataset))

    return dataset, loader
