import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

batch_size = 256
image_size = (224, 224)

train_ds = image_dataset_from_directory('data',
                                        subset='training',
                                        seed=42,
                                        validation_split=0.8,
                                        batch_size=batch_size,
                                        image_size=image_size)

validation_ds = image_dataset_from_directory('data',
                                             subset='validation',
                                             seed=42,
                                             validation_split=0.1,
                                             batch_size=batch_size,
                                             image_size=image_size)

test_ds = image_dataset_from_directory('data',
                                       seed=0,
                                       subset='both',
                                       validation_split=0.1,
                                       batch_size=batch_size,
                                       image_size=image_size)

# переходим к случайной аугументации данных
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


# for images, labels in train_ds.take(1):
#     plt.figure(figsize=(10, 10))
#     first_image = images[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(
#             tf.expand_dims(first_image, 0), training=True
#         )
#         plt.imshow(augmented_image[0].numpy().astype("int32"))
#         plt.title(int(labels[0]))
#         plt.axis("off")
#
#     plt.show()


# инициализируем базовую предобученную на imagenet модель нейросети ResNet50
base_model = ResNet50(weights='imagenet')

# Замораживаем нашу базовую модель
base_model.trainable = False

# Создаем новый верхний выходной слой
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # добавляем рандомную аугментацию данных

# Базовая модель содержит слои пакетной нормы. Мы хотим, чтобы они оставались в режиме вывода
# когда мы разморозим базовую модель для тонкой настройки, поэтому мы убедимся, что base_model здесь работает в режиме вывода.
x = base_model(x, training=False)
# x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Регуляризация с отсевом
outputs = keras.layers.Dense(1)(x)  # создаем выходной слой(dense -просто распростарненный слой)
model = keras.Model(inputs, outputs)

model.summary()  # резюмирование модели(вывод конфигурации)

# далее приступаем к обучению верхнего слоя
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)
epochs = 10  # test various variations
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# Разморозить base_model. Обратите внимание, что он продолжает работать в режиме вывода
# так как мы передали `training=False` при вызове. Это значит, что
# слои batchnorm не будут обновлять свою пакетную статистику.
# Это предотвратит отмену всех тренировочных слоев слоями пакетной нормы.
# мы сделали до сих пор.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)