import os
import torchvision
from torchvision.datasets import ImageFolder

mean_vec = [0.4478, 0.4967, 0.3218]
std_vec = [0.2053, 0.2062, 0.1792]


class CassavaFolder(ImageFolder):

    def __init__(self, root, transform=None):
        super(CassavaFolder, self).__init__(root=root, transform=transform)

    def __getitem__(self, item):
        path, target = self.imgs[item]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (path, img), target
