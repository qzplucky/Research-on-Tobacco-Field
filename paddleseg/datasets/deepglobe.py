# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

import numpy as np
from PIL import Image

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class DeepGlobe(Dataset):
    """
    DeepGlobe dataset (http://deepglobe.org/).

    Note that, the dataset provides train set with annotated images, val set without annotated images,
    test set without annotated images. Therefore, please first split all annotated images to train,
    val and test set.

    The folder structure is as follow:
        deepglobe
        |--train
        |-- train.txt
        |-- val.txt
        |-- test.txt

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int, optional): Number of classes. Default: 2.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes=2,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        super().__init__(transforms, dataset_root, num_classes, mode,
                         train_path, val_path, test_path, separator,
                         ignore_index, edge)
        '''
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge
        self.file_list = list()

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if mode not in ['train', 'val']:
            raise ValueError(
                "mode should be 'train' or 'val' , but got {}.".format(
                    mode))

        img_dir = os.path.join(self.dataset_root, str(mode))
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(img_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(glob.glob(os.path.join(img_dir, '*_mask.png')))
        img_files = sorted(glob.glob(os.path.join(img_dir, '*_sat.jpg')))
        if len(label_files) != len(img_files):
            raise ValueError("The number of train and label images is not the same")

        self.file_list = [[img_path, label_path] \
            for img_path, label_path in zip(img_files, label_files)]
        '''

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            # The label needs to be casted to binary.
            im, label = self.transforms(im=image_path, label=label_path)
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                return im, label
