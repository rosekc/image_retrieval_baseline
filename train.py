import functools
import json
import os
import random
from collections import defaultdict
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.utils import losses_utils

from data_loader import ImageDataLoader
from model import build_baseline_model
from transform import image_crop, image_pad, random_flip, reshape, standardize, suit_for_min_shape


def preprocess_image(img):
    img = img_to_array(img)
    img = image_crop(img, target_shape=(256, 256), random_crop=True)
    img = random_flip(img, True, False)
    img = standardize(img)
    return img


class LearningRateScheduler:
    def __init__(self, optimizers, warm_up=True):
        self.optimizers = optimizers
        self.warm_up = warm_up
        self.init_lrs = [K.get_value(o.lr) for o in optimizers]

    def __call__(self, epoch):
        for init_lr, o in zip(self.init_lrs, self.optimizers):
            lr = K.get_value(o.lr)
            if epoch < 10:
                lr = init_lr * epoch / 10
            elif epoch in (40, 70):
                lr *= 0.1
            K.set_value(o.lr, lr)


class BatchHard(tf.keras.losses.Loss):
    def __init__(self, margin=0.3, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)

        self.margin = margin

        self.running_corrects = tf.keras.metrics.Mean()
        self.running_margin = tf.keras.metrics.Mean()

    def call(self, y_true, y_pred):
        batch_size = tf.cast(tf.shape(y_true)[0], y_pred.dtype)

        diffs = tf.expand_dims(y_pred, axis=1) - \
            tf.expand_dims(y_pred, axis=0)
        dists = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)

        same_label_mask = tf.equal(tf.expand_dims(
            y_true, axis=1), tf.expand_dims(y_true, axis=0))
        neg_mask = tf.math.logical_not(same_label_mask)
        pos_mask = tf.math.logical_xor(
            same_label_mask, tf.eye(batch_size, dtype=tf.bool))

        hardest_pos = tf.reduce_max(
            dists * tf.cast(pos_mask, dists.dtype), axis=1)
        hardest_neg = tf.map_fn(lambda x: tf.reduce_min(
            tf.boolean_mask(x[0], x[1])), (dists, neg_mask), tf.float32)

        diff = hardest_pos - hardest_neg

        loss = tf.nn.relu(diff + self.margin)

        mean_margin = tf.reduce_sum(-diff) / batch_size
        corrects = tf.reduce_sum(
            tf.cast(-diff > self.margin, tf.float32)) / batch_size

        self.running_margin.update_state(mean_margin)
        self.running_corrects.update_state(corrects)
        return loss


def train():
    if tf.__version__.split('.')[0] != '2':
        tf.enable_eager_execution()

    transform = [preprocess_image]

    p = 4
    k = 16
    batch_size = p * k
    learning_rate = 0.05
    epochs = 120
    img_shape = (256, 256)
    margin = 0.3

    data = ImageDataLoader('.', transforms=transform, p=p, k=k)

    class_num = len(data.classes)
    steps_per_epoch = (class_num // p) * 40

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    num_replicas_in_sync = strategy.num_replicas_in_sync

    batch_hard_func = BatchHard(
        margin=margin, reduction=losses_utils.ReductionV2.NONE)
    id_loss_func = SparseCategoricalCrossentropy(
        reduction=losses_utils.ReductionV2.NONE)

    id_loss_metrics = tf.keras.metrics.Mean()
    id_corrects = tf.keras.metrics.SparseCategoricalAccuracy()

    running_corrects = batch_hard_func.running_corrects
    running_margin = batch_hard_func.running_margin
    triple_loss_metrics = tf.keras.metrics.Mean()

    def loss_func(id_output, features, labels):
        triple_loss = tf.reduce_sum(batch_hard_func(labels, features)) / global_batch_size
        id_loss = tf.reduce_sum(id_loss_func(
            labels, id_output)) / global_batch_size
        id_loss_metrics.update_state(id_loss)
        triple_loss_metrics.update_state(triple_loss)
        return id_loss + triple_loss

    with strategy.scope():
        model = build_baseline_model(class_num, img_shape)

        finetune_weights = model.get_layer(name='resnet50').trainable_weights
        finetune_optimizer = SGD(
            learning_rate=learning_rate * 0.1, momentum=0.9, nesterov=True)

        train_weights = [
            w for w in model.trainable_weights if not w in finetune_weights]
        optimizer = SGD(learning_rate=learning_rate,
                        momentum=0.9, nesterov=True)

        all_weights = finetune_weights + train_weights

        # sgd = SGD(learning_rate=1)

        learning_rate_scheduler = LearningRateScheduler(
            [optimizer, finetune_optimizer])

        data_iter = data.flow()

        with open('checkpoint/model.json', 'w', encoding='utf-8') as fp:
            fp.write(model.to_json())

        def train_step(batch):
            imgs, labels = batch

            with tf.GradientTape(persistent=True) as tape:
                id_output, features = model(imgs)

                loss = loss_func(id_output, features, labels)
                # l2_loss = weight_decay * \
                #     tf.add_n([tf.nn.l2_loss(v)
                #               for v in model.trainable_weights])

            grads = tape.gradient(loss, all_weights)
            # l2_grads = tape.gradient(l2_loss, model.trainable_weights)

            finetune_grads = grads[:len(finetune_weights)]
            train_grads = grads[len(finetune_weights):]

            finetune_optimizer.apply_gradients(
                zip(finetune_grads, finetune_weights))
            optimizer.apply_gradients(zip(train_grads, train_weights))
            # sgd.apply_gradients(zip(l2_grads, model.trainable_weights))

            id_corrects.update_state(labels, id_output)

            return loss

        @tf.function
        def distributed_train_step(batch):
            per_replica_losses = strategy.experimental_run_v2(
                train_step, args=(batch,))
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)
            return loss

        # model.load_weights('checkpoint/30.h5')
        # learning_rate_scheduler([optimizer, finetune_optimizer], 20)

        with K.learning_phase_scope(1):
            history = defaultdict(list)

            for cur_epoch in range(1, epochs + 1):
                print('Epoch {}/{}'.format(cur_epoch, epochs))
                progbar = Progbar(steps_per_epoch)

                learning_rate_scheduler(cur_epoch)

                for i in range(steps_per_epoch):
                    batch = next(data_iter)
                    if len(batch[1]) != batch_size:
                        batch = next(data_iter)
                        assert len(batch[1]) == batch_size

                    loss = distributed_train_step(batch)

                    cur_data = [('loss', loss), ('id_acc', id_corrects.result())]

                    progbar.add(1, values=cur_data)

                print(
                    f'acc: {running_corrects.result()} margin: {running_margin.result()}')
                print(
                    f'id acc: {id_corrects.result()} id loss: {id_loss_metrics.result()}')
                print(
                    f'triple_loss: {triple_loss_metrics.result()}')
                running_corrects.reset_states()
                running_margin.reset_states()
                triple_loss_metrics.reset_states()
                id_corrects.reset_states()
                id_loss_metrics.reset_states()

                for key, val in cur_data:
                    history[key].append(float(val))

                with open('checkpoint/history.json', 'w') as fp:
                    json.dump(history, fp)

                if cur_epoch % 5 == 0:
                    model.save_weights(f'checkpoint/{cur_epoch}.h5')


if __name__ == "__main__":
    train()
