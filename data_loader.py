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
from tensorflow.keras.utils import to_categorical

from model import build_model


class ImageIterator(Iterator):
    def __init__(self, n, data, num_classes, transforms, batch_size, shuffle, seed):
        super().__init__(n, batch_size, shuffle, seed)

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

        batch_labels = labels[index_array]
        batch_paths = paths[index_array]

        batch_imgs = self.preprocess_image(batch_paths)

        # batch_labels = to_categorical(
        #     batch_labels, num_classes=self.num_classes)

        return batch_imgs, batch_labels


class ImageDataLoader:
    def __init__(self, root_dir, transforms=None, batch_size=2, shuffle=True, base_loader=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not isinstance(transforms, list):
            transforms = []
        self.transforms = transforms

        self.paths = []
        self.labels = []
        self.cams = []
        self.classes = []

        if base_loader:
            self.class_to_label_id = base_loader.class_to_label_id
            self.classes = base_loader.classes

        self._walk_path(root_dir, base_loader)

        self.labels = np.array(self.labels)
        self.paths = np.array(self.paths)
        self.cams = np.array(self.cams)

        # sorted_idxs = np.argsort(self.paths)

        # self.labels = self.labels[sorted_idxs]
        # self.paths = self.paths[sorted_idxs]
        # self.cams = self.cams[sorted_idxs]

        if not base_loader:
            self.classes.sort()
            self.classes = np.array(self.classes)
            self.class_to_label_id = {
                self.classes[i]: i for i in range(len(self.classes))}

        self.label_ids = np.array([self.class_to_label_id[c]
                                   for c in self.labels])

    def _walk_path(self, root_dir, base_loader):
        vis_class = set()

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if not imghdr.what(full_path) is None:
                    label, cam = self._parse_path(full_path)

                    if base_loader:
                        if not label in self.class_to_label_id:
                            # skip when get unknow label
                            continue
                    elif not label in vis_class:
                        self.classes.append(label)
                        vis_class.add(label)

                    self.paths.append(full_path)
                    self.labels.append(label)
                    self.cams.append(cam)

    def _parse_path(self, path):
        basename = os.path.basename(path)
        spilted = basename.split('_')
        label = int(spilted[0])
        cam = int(spilted[1][1])
        return label, cam

    def __len__(self):
        return len(self.paths)

    def flow(self):
        paths = self.paths
        label_ids = self.label_ids
        combined_data = [label_ids, paths]

        return ImageIterator(len(paths), combined_data, len(self.classes), self.transforms, self.batch_size, self.shuffle, None)

    def __iter__(self):
        return iter(self.flow())


class TripleLossImageIterator(ImageIterator):
    def __init__(self, data, num_classes, transforms, batch_size, shuffle, k, seed):
        n = num_classes
        super().__init__(n, data, num_classes, transforms, batch_size, shuffle, seed)
        self.k = k
    
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
            pos_idx = np.random.choice(pos_idx, self.k)
            pos_paths = paths[pos_idx]
            batch_paths.append(pos_paths)
        
        batch_labels = np.concatenate([np.repeat(i, self.k) for i in batch_labels])
        batch_paths = np.stack(batch_paths).flatten()
        batch_imgs = self.preprocess_image(batch_paths)

        return batch_imgs, batch_labels


class TripleLossImageDataLoader(ImageDataLoader):
    def __init__(self, root_dir, transforms=None, batch_size=2, shuffle=True, k=4):
        super().__init__(root_dir, transforms, batch_size, shuffle)
        self.k = k
    
    def __len__(self):
        return len(self.classes)
    
    def flow(self):
        paths = self.paths
        label_ids = self.label_ids
        combined_data = (label_ids, paths)

        return TripleLossImageIterator(combined_data, len(self.classes), self.transforms, self.batch_size, self.shuffle, self.k, None)

if __name__ == "__main__":
    dg0 = TripleLossImageDataLoader(
        '../Market/bounding_box_train', batch_size=32)
    data_iter = dg0.flow()

    data = next(data_iter)
    data[0][0][0].save('test.jpg')
