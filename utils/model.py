import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



class Model:
    def __init__(self, model_name, numclasses, img_height=224, img_width=224):
        if model_name.lower() == "base":
            self.net = self._base_model(img_height, img_width, numclasses)
        elif model_name.lower() == "eff_b0":
            self.net = self._base_model(img_height, img_width, numclasses)
        else:
            print("Wrong model name.")
            exit(-1)
                

    @staticmethod
    def _base_model(img_height, img_width, numclasses):
        # model = Sequential([
        #     layers.experimental.preprocessing.Rescaling(
        #         1./255, input_shape=(img_height, img_width, 3)),
        #     layers.Conv2D(16, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(32, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(64, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Flatten(),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dense(numclasses)
        # ])

        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(numclasses))

        model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


        model.summary()

        return model
