import imghdr
import os
from collections import defaultdict
from functools import reduce
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img, Iterator)

from transform import (image_crop, image_pad, random_flip,
                       random_horizontal_flip, reshape, standardize,
                       suit_for_min_shape)


class ImageIterator(Iterator):
    def __init__(self, data, batch_size, transforms, shuffle, seed):
        super().__init__(len(data[0]), batch_size, shuffle, seed)

        self.data = data
        self.transforms = transforms

    def preprocess_image(self, paths):
        imgs = map(load_img, paths)
        imgs = list(map(lambda x: reduce(lambda v, func: func(
            v), self.transforms, x), imgs))
        if isinstance(imgs[0], np.ndarray):
            imgs = np.asarray(imgs)
        return imgs

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        labels, paths = self.data

        batch_labels = labels[index_array]
        batch_paths = paths[index_array]

        batch_imgs = self.preprocess_image(batch_paths)

        return batch_imgs, batch_labels


class QueryImageIterator(ImageIterator):
    def preprocess_image(self, paths, crops):
        imgs = map(load_img, paths)
        imgs = list(map(lambda x: reduce(lambda v, func: func(
            v), self.transforms, x), imgs))

        cropped_imgs = []

        for i, c in zip(imgs, crops):
            left, top, right, bottom = c
            img = i[left:right, top:bottom, :]
            img = image_crop(img, target_shape='largest_square', crop_mode='center')
            img = reshape(img, (256, 256))
            img = standardize(img)

            cropped_imgs.append(img)

        cropped_imgs = np.asarray(cropped_imgs)

        return cropped_imgs

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        labels, paths, crops = self.data

        batch_labels = labels[index_array]
        batch_paths = paths[index_array]
        batch_crops = crops[index_array]

        batch_imgs = self.preprocess_image(batch_paths, batch_crops)

        return batch_imgs, batch_labels


class TripleLossImageIterator(Iterator):
    def __init__(self, data, p, k, num_classes, transforms, shuffle, seed):
        super().__init__(num_classes, p, shuffle, seed)

        self.p = p
        self.k = k

        self.num_classes = num_classes
        self.data = data
        self.transforms = transforms

    def preprocess_image(self, paths):
        imgs = map(load_img, paths)
        imgs = list(map(lambda x: reduce(lambda v, func: func(
            v), self.transforms, x), imgs))
        if isinstance(imgs[0], np.ndarray):
            imgs = np.array(imgs)
        return imgs

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        labels, paths = self.data

        batch_labels = index_array
        batch_paths = []

        for i in range(len(batch_labels)):
            cur_label = batch_labels[i]

            pos_idx = np.argwhere(cur_label == labels).flatten()
            pos_idx = np.random.choice(pos_idx, self.k, replace=False)
            pos_paths = paths[pos_idx]
            batch_paths.append(pos_paths)

        batch_labels = np.concatenate(
            [np.repeat(i, self.k) for i in batch_labels])
        batch_paths = np.stack(batch_paths).flatten()
        batch_imgs = self.preprocess_image(batch_paths)

        return batch_imgs, batch_labels


class ImageDataLoader:
    def __init__(self, root_dir, name='train', transforms=None, p=6, k=4, shuffle=True):
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.shuffle = shuffle
        self.name = name

        if not isinstance(transforms, list):
            transforms = []
        self.transforms = transforms

        self.paths = np.load(os.path.join(root_dir, f'{name}_paths.npy'))
        self.labels = np.load(os.path.join(root_dir, f'{name}_labels.npy'))
        if name == 'query':
            self.crops = np.load(os.path.join(root_dir, 'query_crops.npy'))

        self.classes = np.unique(self.labels)

        self.classes = np.sort(self.classes)
        self.class_to_label_id = {
            self.classes[i]: i for i in range(len(self.classes))}

        self.label_ids = np.array([self.class_to_label_id[c]
                                   for c in self.labels])

    def __len__(self):
        return len(self.paths)

    def flow(self):
        paths = self.paths
        label_ids = self.label_ids
        combined_data = [label_ids, paths]

        return TripleLossImageIterator(combined_data, self.p, self.k, len(self.classes), self.transforms, self.shuffle, None)

    def val_flow(self):
        paths = self.paths
        label_ids = self.label_ids
        combined_data = [label_ids, paths]
        if self.name == 'query':
            combined_data.append(self.crops)
            return QueryImageIterator(combined_data, self.batch_size, self.transforms, self.shuffle, None)

        return ImageIterator(combined_data, self.batch_size, self.transforms, self.shuffle, None)

    def __iter__(self):
        return iter(self.flow())


if __name__ == "__main__":
    dg0 = TripleLossImageDataLoader(
        '../Market/bounding_box_train', batch_size=32)
    data_iter = dg0.flow()

    data = next(data_iter)
    data[0][0][0].save('test.jpg')
