import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


__all__ = ["test_model", 'EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
           'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

__models__ = ["test_model", "eff_b0", "eff_b1", "eff_b2",
              "eff_b3", "eff_b4", "eff_b5", "eff_b6", "eff_b7"]


class Model:
    def __init__(self, model_name, numclasses, img_height=224, img_width=224, \
                train_corenet=False):
        if model_name.lower() not in __models__:
            print("Wrong model name. 'model_name' should be one of {}".format(__models__))
            exit(-1)
        if model_name.lower() == "test_model":
            self.net = self._base_model(img_height, img_width, numclasses, train_corenet=train_corenet)
        elif model_name.lower() == "eff_b0":
            self.net = self._effb0_model(img_height, img_width, numclasses, train_corenet=train_corenet)
        else:
            print("Wrong model name.")
            exit(-1)

    @staticmethod
    def _base_model(img_height, img_width, numclasses, train_corenet=False):
        model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(numclasses)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        model.summary()

        return model

    @staticmethod
    def _effb0_model(img_height, img_width, numclasses, train_corenet=False):
        core_net = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False, weights='imagenet')

        core_net.trainable = train_corenet

        inputs = keras.Input(shape=(224, 224, 3))
        x = core_net(inputs, training=train_corenet)

        # ------- Adding Classification Head ---------
        # add a global spatial average pooling layer
        x = layers.GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = layers.Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = layers.Dense(2, activation='softmax')(x)

        # this is the model we will train
        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        model.summary()

        return model
