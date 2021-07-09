from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Dataset:
    def __init__(self, train_dir, valid_dir, image_size=(224, 224), batch_size=32, NUM_TRAIN=20000, NUM_VALID=5000, prefetch=500, seed=123):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.prefetch = prefetch

    def get_ds(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            interpolation='bilinear',
        )

        valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.valid_dir,
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            interpolation='bilinear',
        )

        if self.prefetch > 0:
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(self.prefetch).prefetch(buffer_size=AUTOTUNE)
            valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, valid_ds

    def get_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        # Note that the validation data should not be augmented!
        valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            directory=self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            color_mode="rgb",
            class_mode="sparse",
            shuffle=True,
            seed=42,
        )

        valid_generator = valid_datagen.flow_from_directory(
            directory=self.valid_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            color_mode="rgb",
            class_mode="sparse",
            shuffle=True,
            seed=42,
        )

        return train_generator, valid_generator

    @ staticmethod
    def load_image_train(image_file, label):
        image, label = load(image_file, label)
        image = random_jitter(image)
        image = normalize(image)
        return image, label

    @ staticmethod
    def load_image_val(image_file, label):
        image, label = load(image_file, label)
        image = central_crop(image)
        image = normalize(image)
        return image, label

    @ staticmethod
    def load(f, label):
        # load the file into tensor
        image = tf.io.read_file(f)
        # Decode it to JPEG format
        image = tf.image.decode_jpeg(image)
        # Convert it to tf.float32
        image = tf.cast(image, tf.float32)

        return image, label

    @ staticmethod
    def resize(input_image, size):
        return tf.image.resize(input_image, size)

    @ staticmethod
    def random_crop(input_image):
        return tf.image.random_crop(input_image, size=[150, 150, 3])

    @ staticmethod
    def central_crop(input_image):
        image = resize(input_image, [176, 176])
        return tf.image.central_crop(image, central_fraction=0.84)

    @ staticmethod
    def random_rotation(input_image):
        angles = np.random.randint(0, 3, 1)
        return tf.image.rot90(input_image, k=angles[0])

    @ staticmethod
    def random_jitter(input_image):
        # Resize it to 176 x 176 x 3
        image = resize(input_image, [176, 176])
        # Randomly Crop to 150 x 150 x 3
        image = random_crop(image)
        # Randomly rotation
        image = random_rotation(image)
        # Randomly mirroring
        image = tf.image.random_flip_left_right(image)
        return image

    @ staticmethod
    def normalize(input_image):
        mid = (tf.reduce_max(input_image) + tf.reduce_min(input_image)) / 2
        input_image = input_image / mid - 1
        return input_image
