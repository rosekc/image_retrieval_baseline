import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Dense, Dropout, Flatten, Input, Lambda,
                                     LeakyReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def build_baseline_model(class_number, img_shape):
    img = Input(shape=(*img_shape, 3))

    resnet_model = ResNet50(
        include_top=False, pooling='avg', weights='imagenet', input_tensor=img)
    resnet_output = resnet_model(img)
    
    feature_output = BatchNormalization(center=False)(resnet_output)

    classify_output = Dense(class_number, activation='softmax')(feature_output)

    return Model(img, [classify_output, feature_output])