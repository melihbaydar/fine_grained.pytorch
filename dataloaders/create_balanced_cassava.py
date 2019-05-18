import os
import numpy as np
import shutil


def create_balanced(train_dir, subset_dir):
    class_names = sorted(list(os.listdir(train_dir)))
    print(class_names)
    class_freq = [0] * len(class_names)
    # get train set class image frequencies
    for i, class_name in enumerate(class_names):
        class_freq[i] = len(os.listdir(
            os.path.join(train_dir, class_name)))
    print("Class freq in train set")
    for i in range(len(class_names)):
        print(class_names[i], " - ", class_freq[i])
    min_freq = min(class_freq)
    for i, class_name in enumerate(class_names):
        class_image_names = os.listdir(os.path.join(train_dir, class_name))
        subset_class_dir = os.path.join(cassava_subset_dir, class_name)
        # create subset class directory if not exists
        if not os.path.isdir(subset_class_dir):
            os.mkdir(subset_class_dir)
        np.random.seed(12)
        # get random images from class class_name
        subset_class_indices = np.random.permutation(class_freq[i])[:min_freq]
        for image_index in subset_class_indices:
            image2copy = os.path.join(
                train_dir, class_name, class_image_names[image_index]
            )
            shutil.copy(image2copy, subset_class_dir)
    subset_class_freq = [0] * len(class_names)
    for i, class_name in enumerate(class_names):
        subset_class_freq[i] = len(os.listdir(
            os.path.join(cassava_subset_dir, class_name)))
    print("Class freq in subset")
    for i in range(len(class_names)):
        print(class_names[i], " - ", subset_class_freq[i])


if __name__ == '__main__':
    cassava_train_dir = '../cassava/train'
    cassava_subset_dir = '../cassava/subset'
    if not os.path.isdir(cassava_subset_dir):
        os.mkdir(cassava_subset_dir)
    create_balanced(cassava_train_dir, cassava_subset_dir)
