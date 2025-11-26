#!/usr/bin/env python3
# shebang line for linux / mac

import glob
from random import randint
import random
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse
from dataset import Dataset

from torchvision import transforms


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='/home/hogu/Desktop/SAVI-Hugo-Sachim_108510-25-26/savi_datasets/mnist/')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Create datasets
    # ------------------------------------
    dataset = Dataset(args, is_train=True)

    # ------------------------------------
    # Ex1 c)
    # ------------------------------------
    # call getitem for an idx and print the resutl
    image_tensor, label_tensor = dataset.__getitem__(87)  # type: ignore

    print('Image tensor shape: ' + str(image_tensor.shape))
    print('Label tensor: ' + str(label_tensor))

    # Display the image
    to_pil = transforms.ToPILImage()
    image = to_pil(image_tensor)  # get the image from the tensor

    plt.figure()
    plt.imshow(image, cmap='gray')

    # Get teh value of the digit to put on the title
    label = label_tensor.tolist()  # get the value from the tensor
    # print('Label = ' + str(label))
    # max_value = max(label)
    # print('Max value = ' + str(max_value))
    # max_index = label.index(max_value)
    # print('Max index = ' + str(max_index))

    label_name = label.index(max(label))
    plt.title('Label ' + str(label_name))

    plt.axis("off")
    plt.show()

    # ------------------------------------
    # Ex1 d)
    # ------------------------------------
    # Show a mosaic of images
    # Homework


    # --- Ex1 d) ---
    # Criar um mosaico 5x5 de imagens aleatórias do dataset

    to_pil = transforms.ToPILImage()

    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    for r in range(rows):
        for c in range(cols):
            idx = random.randint(0, len(dataset) - 1)
            image_tensor, label_tensor = dataset.__getitem__(idx)

            print('Image tensor shape: ' + str(image_tensor.shape))
            print('Label tensor: ' + str(label_tensor))

            image = to_pil(image_tensor)
            label = label_tensor.tolist()
            label_name = label.index(max(label))

            axes[r][c].imshow(image, cmap='gray')
            axes[r][c].set_title(str(label_name))
            axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
