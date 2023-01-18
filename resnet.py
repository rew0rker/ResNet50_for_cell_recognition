import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory


batch_size = 256
image_size = (875, 656)

train_ds = image_dataset_from_directory('data',
                                        subset='training',
                                        seed=42,
                                        validation_split=0.7,
                                        batch_size=batch_size,
                                        image_size=image_size)

validation_ds = image_dataset_from_directory('data',
                                             subset='validation',
                                             seed=42,
                                             validation_split=0.1,
                                             batch_size=batch_size,
                                             image_size=image_size)

test_ds = image_dataset_from_directory('data',
                                       seed=42,
                                       subset='both',
                                       validation_split=0.2,
                                       batch_size=batch_size,
                                       image_size=image_size)


# size = (224, 224)  ## зададим размер всех изображений на 224x224 и применим этот размер ко всем датасетам
# train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
