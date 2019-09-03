import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
import numpy as np
from model import build_model
from data_loader import ImageDataLoader
from transform import random_horizontal_flip
import cv2


def reshape(img):
    img = img.resize((224, 224))
    return img


def build_feature_model(model):
    input_tensor = model.input

    blockneck_output = model.get_layer('batch_normalization').output

    # x = tf.add(tf.norm(blockneck_output, ord=2, axis=1, keepdims=True), 1e-8)
    # x = tf.divide(blockneck_output, x)

    feature_model = Model(input_tensor, blockneck_output)

    return feature_model


def extract_feature(feature_model, data_loader):
    data_iter = data_loader.flow()

    # features = base_model.predict_generator(data_iter)
    features = None

    for batch in data_iter:
        batch_imgs = batch[0]
        batch_features = tf.zeros((len(batch_imgs), *feature_model.output.shape[1:]))

        for flip in [0, 1]:
            if flip:
                imgs = np.array([cv2.flip(i, flipCode=1) for i in batch_imgs])
            else:
                imgs = batch_imgs

            batch_features += feature_model(imgs)
        
        # batch_features_norm = tf.add(tf.norm(batch_features, ord=2, axis=1, keepdims=True), 1e-8)
        # batch_features = tf.divide(batch_features, batch_features_norm)

        if features is None:
            features = batch_features
        else:
            features = tf.concat([features, batch_features], 0)
        print('{}/{}'.format(data_iter.total_batches_seen, len(data_iter)))
        if data_iter.total_batches_seen == len(data_iter):
            break

    return features


def extract_feature_to_file(path, name, model):
    data_loader = ImageDataLoader(path, transforms=transform, batch_size=64, shuffle=False)

    features = extract_feature(model, data_loader)
    np.save(f'{name}_features.npy', features.numpy())
    np.save(f'{name}_labels.npy', data_loader.labels)
    np.save(f'{name}_cams.npy', data_loader.cams)


if __name__ == "__main__":
    from model import build_identification_model, build_triple_loss_model, build_baseline_model
    from train_eager import val_preprocess_image

    tf.enable_eager_execution()

    with K.learning_phase_scope(0):
        transform = [val_preprocess_image]
        model = build_baseline_model(751, (256, 128))
        model.load_weights('checkpoint/120.h5')
        feature_model = build_feature_model(model)
        extract_feature_to_file('../Market/query', 'query', feature_model)
        extract_feature_to_file(
            '../Market/bounding_box_test', 'gallery', feature_model)
