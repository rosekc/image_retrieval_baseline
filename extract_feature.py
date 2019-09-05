import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)

from data_loader import ImageDataLoader
from transform import (image_crop, image_pad, random_flip,
                       random_horizontal_flip, reshape, standardize,
                       suit_for_min_shape)


def build_feature_model(model):
    input_tensor = model.input

    blockneck_output = model.get_layer('batch_normalization').output

    # x = tf.add(tf.norm(blockneck_output, ord=2, axis=1, keepdims=True), 1e-8)
    # x = tf.divide(blockneck_output, x)

    feature_model = Model(input_tensor, blockneck_output)

    return feature_model


def extract_feature(feature_model, data_loader):
    data_iter = data_loader.val_flow()

    # features = base_model.predict_generator(data_iter)
    features = None

    for batch in data_iter:
        batch_imgs = np.array(batch[0])
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


def extract_feature_to_file(path, name, model, transform):
    data_loader = ImageDataLoader(path, name=name, transforms=transform, shuffle=False)

    features = extract_feature(model, data_loader)
    np.save(f'{name}_features.npy', features.numpy())

def val_preprocess_image(img):
    img = img_to_array(img)
    img = image_crop(img, target_shape='largest_square', crop_mode='center')
    img = reshape(img, (256, 256))
    img = standardize(img)
    return img

def query_preprocess_image(img):
    return img_to_array(img)


if __name__ == "__main__":
    from model import build_baseline_model

    tf.enable_eager_execution()

    with K.learning_phase_scope(0):
        model = build_baseline_model(11, (256, 256))
        model.load_weights('checkpoint/120.h5')
        feature_model = build_feature_model(model)
        extract_feature_to_file('.', 'query', feature_model, [query_preprocess_image])
        extract_feature_to_file('.', 'gallery', feature_model, [val_preprocess_image])
