import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


__all__ = ['EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
           'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', "test_model"]

__models__ = ["eff_b0", "eff_b1", "eff_b2",
              "eff_b3", "eff_b4", "eff_b5", "eff_b6", "eff_b7", "test_model"]


class Model:
    def __init__(self, model_name, numclasses, img_height=224, img_width=224,
                 train_corenet=False):
        self.model_name = model_name
        if model_name.lower() not in __models__:
            print("Wrong model name. 'model_name' should be one of {}".format(__models__))
            exit(-1)
        if model_name.lower() == "test_model":
            self.net = self._test_model(
                img_height, img_width, numclasses, train_corenet=train_corenet)
        elif "eff_b" in model_name.lower():
            self.net = self.effbx_model(
                img_height, img_width, numclasses, train_corenet=train_corenet)
        else:
            print("Wrong model name.")
            exit(-1)

    @staticmethod
    def _test_model(img_height, img_width, numclasses, train_corenet=False):
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                  padding='same', input_shape=(img_height, img_width, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(numclasses, activation='softmax'))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=False),
                      metrics=['accuracy'])
        model.summary()

        return model

    def effbx_model(self, img_height, img_width, numclasses, train_corenet=False):
        core_net = self._get_corenet(self.model_name, weights='imagenet')
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

    @staticmethod
    def _get_corenet(model_name, weights='imagenet'):
        if model_name == __models__[0]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB0(
                include_top=False, weights=weights)
        elif model_name == __models__[1]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB1(
                include_top=False, weights=weights)
        elif model_name == __models__[2]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB2(
                include_top=False, weights=weights)
        elif model_name == __models__[3]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB3(
                include_top=False, weights=weights)
        elif model_name == __models__[4]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB4(
                include_top=False, weights=weights)
        elif model_name == __models__[5]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB5(
                include_top=False, weights=weights)
        elif model_name == __models__[6]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB6(
                include_top=False, weights=weights)
        elif model_name == __models__[7]:
            core_net = tf.keras.applications.efficientnet.EfficientNetB7(
                include_top=False, weights=weights)
        else:
            print("Wrong Model Name {} entered.".format(model_name))
            exit(-1)

        return core_net
