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


def train_with_tfdata(dbds, model_name, num_classes, epochs, train_corenet=False):
    # -----------------------------------------------------------
    #          TF Dataset
    # -----------------------------------------------------------
    train_ds, valid_ds = dbds.get_ds()

    if model_name.lower() == "test_model" and train_corenet:
        train_corenet = False
        print("For '{}', parameter 'trainable' cannot be True. Changing iot to False".format(
            model_name))

    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    #     1./255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]

    mymodel = Model(model_name, num_classes, train_corenet=train_corenet)

    history = mymodel.net.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
    )

    return history


def train_with_generators(dbds, model_name, num_classes, epochs, train_corenet=False):
    # -----------------------------------------------------------
    #           ImageDataGenerator
    # -----------------------------------------------------------

    train_generator, valid_generator = dbds.get_generators()
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

    print(STEP_SIZE_TRAIN, STEP_SIZE_VALID)

    if model_name.lower() == "test_model" and train_corenet:
        train_corenet = False
        print("For '{}', parameter 'trainable' cannot be True. Changing iot to False".format(
            model_name))

    mymodel = Model(model_name, num_classes, train_corenet=train_corenet)
    train_net = mymodel.net

    history = train_net.fit(train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=epochs,
                            use_multiprocessing=True,
                            workers=12,
                            )

    return history


def main():
    print("Tensorflow: v{}".format(tf.__version__))

    data_dir = "/dataset/dogs_cats/train_data"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    image_size = (224, 224)
    batch_size = 256
    epochs = 3
    num_classes = 2
    NUM_TRAIN = 20000
    NUM_VALID = 5000

    # Model name can be 
    # "eff_b0", "eff_b1", "eff_b2", "eff_b3", 
    # "eff_b4", "eff_b5", "eff_b6", "eff_b7", "test_model"
    model_name = "eff_b6"          
    train_corenet = False

    data_api = "keras_gen"         # "tf_data" or "keras_gen"

    dbds = Dataset(train_dir,
                   valid_dir,
                   image_size,
                   batch_size,
                   NUM_TRAIN=20000,
                   NUM_VALID=5000,
                   prefetch=1000)

    if data_api == "tf_data":
        history = train_with_tfdata(
            dbds, model_name, num_classes, epochs, train_corenet)
    elif data_api == "keras_gen":
        history = train_with_generators(dbds, model_name, num_classes, epochs)
    else:
        print("Wrong Input")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Training time: {}".format(time.time() - start_time))
