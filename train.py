import os
import time
import re
from glob import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils.ipp import Dataset
from utils.model import Model


def train_with_tfdata(dbds, model_name, num_classes, epochs):
    # -----------------------------------------------------------
    #          TF Dataset
    # -----------------------------------------------------------
    train_ds, valid_ds = dbds.get_ds()

    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]

    mymodel = Model(model_name, num_classes)

    history = mymodel.net.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs)

    return history


def train_with_generators(dbds, model_name, num_classes, epochs):
    # -----------------------------------------------------------
    #           ImageDataGenerator
    # -----------------------------------------------------------

    train_generator, valid_generator = dbds.get_generators()
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

    print(STEP_SIZE_TRAIN, STEP_SIZE_VALID)

    mymodel = Model(model_name, num_classes)
    train_net = mymodel.net

    history = train_net.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=valid_generator,
                                      validation_steps=STEP_SIZE_VALID,
                                      epochs=epochs,
                                      verbose=1,
                                      use_multiprocessing=True,
                                      workers=4,
                                      )

    return history


def main():
    print("Tensorflow: v{}".format(tf.__version__))

    data_dir = "/dataset/dogs_cats/train_data"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    image_size = (224, 224)
    batch_size = 32
    epochs = 10
    num_classes = 2
    NUM_TRAIN = 20000
    NUM_VALID = 5000
    model_name = "base"

    data_api = "keras_gen"         # "tf_data" or "keras_gen"

    dbds = Dataset(train_dir,
                   valid_dir,
                   image_size,
                   batch_size,
                   prefetch=1000)

    if data_api == "tf_data":
        history = train_with_tfdata(dbds, model_name, num_classes, epochs)
    elif data_api == "keras_gen":
        history = train_with_generators(dbds, model_name, num_classes, epochs)
    else:
        print("Wrong Input")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Training time: {}".format(time.time() - start_time))
