import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision.datasets import ImageFolder

mean_vec = [0.4478, 0.4967, 0.3218]
std_vec = [0.2053, 0.2062, 0.1792]


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


# Adapted from pytorch folder.py that contains DatasetFolder and ImageFolder
def make_dataset(root, paths_dict, class_to_idx, extensions):
    """

    :param root: root folder of given dataset split
    :param paths_dict: contains class names as keys and image paths as values
    :param class_to_idx: class to index dictionary
    :param extensions: extensions to distinguish images
    :return: images as tuple in form (path, class id)
    """
    images = []
    root = os.path.join(root, 'train')
    root = os.path.expanduser(root)
    for target in sorted(class_to_idx.keys()):
        class_dir = os.path.join(root, target)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(paths_dict[target]):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(class_dir, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images


def _create_paths_dict(root_dir, split, split_percentage):
    """

    :param root_dir:
    :param split: split type 'train' or 'val
    :param split_percentage:
    :return paths_dict: a dictionary consisting of class names as
                        keys and image paths as a list for each key
    """
    split_dir = os.path.join(root_dir, 'train')  # get both train and validation data from train dir
    class_names = sorted(os.listdir(split_dir))
    paths_dict = dict()
    total = 0
    for class_name in class_names:
        # for consistent random split generation between runs, set seed
        np.random.seed(12)
        class_dir = os.path.join(split_dir, class_name)
        image_names = sorted(os.listdir(class_dir))
        num_images = len(image_names)
        total += num_images

        all_indices = np.array(np.random.permutation(range(num_images)))
        # if split == 'val':
        #     split_percentage = 1 - split_percentage
        split_border = np.int(num_images * split_percentage)
        split_ind = all_indices[:split_border] if split == 'train' else all_indices[split_border:]
        split_paths = [os.path.join(class_dir, image_names[i]) for i in split_ind]
        paths_dict[class_name] = split_paths
    return paths_dict


def _compute_class_weights(paths_dict, class_to_idx):
    y_train = []
    for class_name in sorted(paths_dict.keys()):
        y_train.extend([class_to_idx[class_name]] * len(paths_dict[class_name]))
    class_weights = compute_class_weight('balanced', np.unique(y_train),  y_train)
    return class_weights


class CassavaTestFolder(ImageFolder):

    def __init__(self, root, transform=None):
        super(CassavaTestFolder, self).__init__(root=root, transform=transform)

    def __getitem__(self, item):

        path, target = self.imgs[item]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, img, target


class CassavaFolder(ImageFolder):

    def __init__(self, root, split, split_percentage=0.8, transform=None):
        """
        A variation of ImageFolder class of pytorch adapted to cassava dataset
        :param root: root directory of cassava dataset that contains train-test-extraimages splits
        :param split: 'train', 'val' or 'extraimages'
        :param split_percentage: 'used only for 'train' and 'val' sets if needed
        :param transform: a series of pytorch transforms to apply to images
        """
        super(CassavaFolder, self).__init__(root=root, transform=transform)
        if split in ['train', 'val']:
            paths_dict = _create_paths_dict(root, split, split_percentage)
        else:  # extraimages
            extraimages_threshold = 0.99
            extraimages_json = 'extraimages_above_threshold_{}.json'.format(extraimages_threshold)
            extraimages_json = os.path.join(root, extraimages_json)
            with open(extraimages_json, 'r') as fp:
                paths_dict = json.load(fp)
        self.classes = sorted(paths_dict.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.samples = self.imgs = make_dataset(root, paths_dict, self.class_to_idx, self.extensions)
        self.class_weights = _compute_class_weights(paths_dict, self.class_to_idx)

    def __getitem__(self, item):

        path, target = self.imgs[item]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, img, target
