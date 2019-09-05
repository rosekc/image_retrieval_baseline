import os
from collections import defaultdict
import numpy as np
import math


querys = defaultdict(list)
image_paths = defaultdict(list)

root_dir = '../../data/paris'
gt_dir  = os.path.join(root_dir, 'gt')
extension = 'jpg'

def parse_path(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    label, idx, level = basename.split('_')
    return label, idx, level

def parse_filename(path):
    basename = os.path.basename(path)
    _, img_label, img_id = basename.split('_')
    return img_label, img_id

for root, dirs, files in os.walk(gt_dir):
    for f in files:
        full_path = os.path.join(gt_dir, f)
        label, idx, level = parse_path(f)

        if not level in ('query', 'good', 'ok'):
            continue
        with open(full_path, 'r') as fp:
            for line in fp.read().splitlines():
                if level == 'query':
                    img_name, left, top, right, bottom = line.split(' ')
                    img_label, img_id = parse_filename(img_name)
                    img_path = f'{root_dir}/{img_label}/{img_name}.{extension}'
                    left, top, right, bottom = (math.floor(float(i))for i in (left, top, right, bottom))
                    querys[label].append((img_path, left, top, right, bottom))
                    continue

                img_label, img_id = parse_filename(line)
                img_path = f'{root_dir}/{img_label}/{line}.{extension}'
                image_paths[label].append(img_path)

query_paths = []
query_crops = []
query_labels = []

for k, v in querys.items():
    for i in v:
        img_path, left, top, right, bottom = i
        query_paths.append(img_path)
        query_crops.append(np.array([left, top, right, bottom]))
        query_labels.append(k)

query_paths = np.array(query_paths)
query_crops = np.stack(query_crops)
query_labels = np.array(query_labels)

np.save('query_paths', query_paths)
np.save('query_crops', query_crops)
np.save('query_labels', query_labels)

train_rate = 0.8

train_paths = []
train_labels = []
gallery_paths = []
gallery_labels = []

for k, v in image_paths.items():
    v = np.asarray(v)
    path_len = len(v)
    train_len = int(path_len * train_rate)
    gallery_len = path_len - train_len
    train_idx = np.random.choice(path_len, train_len, replace=False)
    train_mask = np.zeros(path_len, dtype=np.bool)
    train_mask[train_idx] = True

    train_paths.append(v[train_mask])
    gallery_paths.append(v[np.logical_not(train_mask)])

    train_labels.append(np.repeat(k, train_len))
    gallery_labels.append(np.repeat(k, gallery_len))

train_paths = np.concatenate(train_paths)
train_labels = np.concatenate(train_labels)
gallery_paths = np.concatenate(gallery_paths)
gallery_labels = np.concatenate(gallery_labels)

mask = np.isin(train_paths, query_paths, invert=True)
train_paths = train_paths[mask]
train_labels = train_labels[mask]

mask = np.isin(gallery_paths, query_paths, invert=True)
gallery_paths = gallery_paths[mask]
gallery_labels = gallery_labels[mask]

np.save('train_paths', train_paths)
np.save('train_labels', train_labels)
np.save('gallery_paths', gallery_paths)
np.save('gallery_labels', gallery_labels)
print(len(train_paths))


