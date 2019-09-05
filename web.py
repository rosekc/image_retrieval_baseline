from flask import Flask, render_template
from model import build_baseline_model
import numpy as np
from data_loader import ImageDataLoader
from extract_feature import val_preprocess_image, query_preprocess_image

app = Flask(__name__)
app.static_folder = '../../data'
app.template_folder = '.'


def load_data(name):
    return np.load(f'{name}.npy')


query_data_loader = ImageDataLoader(
    '.', transforms=[query_preprocess_image], name='query', shuffle=False)
gallery_data_loader = ImageDataLoader(
    '.', transforms=[val_preprocess_image], name='gallery', shuffle=False)

query_paths = query_data_loader.paths
gallery_paths = gallery_data_loader.paths

query_features = load_data('query_features')
query_labels = load_data('query_labels')

gallery_features = load_data('gallery_features')
gallery_labels = load_data('gallery_labels')


def parse_path(path):
    return path[len('../../data/'):]


@app.route('/')
def index():
    query_idx = np.random.randint(0, len(query_paths), 1)[0]
    query_path = query_paths[query_idx]
    query_feature = query_features[query_idx]
    query_label = query_labels[query_idx]

    diffs = np.reshape(query_feature, (1, 1, -1)) - \
        np.expand_dims(gallery_features, axis=0)
    distances = np.sqrt(np.sum(np.square(diffs), axis=-1) + 1e-12)
    distance_rank_index = np.argsort(distances)

    same_label_index = np.argwhere(query_label == gallery_labels).flatten()

    good_index = same_label_index

    rank_index = distance_rank_index.flatten()

    is_good = np.isin(rank_index[:10], good_index)

    return render_template('template.html', query_path=parse_path(query_path), gallery_paths=[parse_path(p) for p in gallery_paths[rank_index[:10]]], is_good=is_good, zip=zip)
