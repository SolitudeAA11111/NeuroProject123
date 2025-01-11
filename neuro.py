#neuro
import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imutils
import re
import natsort

# Функция для создания новых классов
def class_maker(classes):
    print(classes)
    unique_classes = sorted(set(classes))  # Уникальные и отсортированные классы
    new_to_old_mapping = {new: old for new, old in enumerate(unique_classes)}  # Новый класс -> Старый класс
    old_to_new_mapping = {old: new for new, old in new_to_old_mapping.items()}  # Старый класс -> Новый класс
    new_classes = [old_to_new_mapping[c] for c in classes]  # Преобразуем старые классы в новые
    print(new_classes)
    return new_to_old_mapping, new_classes

# Функция для загрузки датасета
def load_dataset(folder, cache_file="preprocessed_data.npz"):
    # Проверяем, существует ли файл с кешированными данными
    if os.path.exists(cache_file):
        print(f"Загрузка предобработанных данных из {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return data['images'], data['classes'], data['inverse_mapping'].item()

    images = []
    classes = []
    filenames = natsort.natsorted(os.listdir(folder))
    for filename in filenames:
        if filename.endswith(('.png', '.jpg', '.JPG', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                img = rgb_img_to_gray(img)
                images.append(img)
                classes.append(int(re.search(r'(\d+)-\d+', img_path).group(1)))
                print(f"Загружено изображение: {filename}")
            except Exception as e:
                print(f"Не удалось загрузить изображение {filename}: {e}")

    # Генерация новых классов
    new_to_old_mapping, new_classes = class_maker(classes)

    # Сохранение предобработанных данных в файл
    print(f"Сохранение предобработанных данных в {cache_file}...")
    np.savez(cache_file, images=images, classes=new_classes, inverse_mapping=new_to_old_mapping)

    return images, new_classes, new_to_old_mapping

#Старый
def rgb_img_to_gray(frame):
    frame = np.array(frame)
    frame = cv2.resize(frame, (512,512) )
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO_INV)[1]
    return gray

# Функция для аугментации данных
def augmentation(img_height=512, img_width=512):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(
            img_height, img_width, 1)),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomZoom(0.1),
    ])
    return data_augmentation


# Функция для создания тренировочных и тестовых данных
def create_test_train_data(data, y):
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.1)
    x_train = tf.stack(x_train)
    x_test = tf.stack(x_test)
    y_train = tf.stack(y_train)
    y_test = tf.stack(y_test)
    return x_train, x_test, y_train, y_test

# Функция для создания модели
def create_model(N, config):

    print("model")
    model_config = config['model']
    model_layers = [
        augmentation(),
        layers.Rescaling(1./255, input_shape=(model_config['img_height'], model_config['img_width'], 1))
    ]

    for layer in model_config['layers']:
        if layer['type'] == 'Conv2D':
            model_layers.append(layers.Conv2D(layer['filters'], layer['kernel_size'], padding=layer['padding'], activation=layer['activation']))
        elif layer['type'] == 'MaxPooling2D':
            model_layers.append(layers.MaxPooling2D(pool_size=layer['pool_size']))
        elif layer['type'] == 'Flatten':
            model_layers.append(layers.Flatten())
        elif layer['type'] == 'Dense':
            # units = layer['units'] if isinstance(layer['units'], int) else eval(layer['units'])
            model_layers.append(layers.Dense(N, activation=layer['activation']))

    model = keras.Sequential(model_layers)

    print("compile")
    compile_config = config['compile']
    model.compile(
        optimizer=getattr(tf.keras.optimizers, compile_config['optimizer']['type'])(learning_rate=compile_config['optimizer']['learning_rate']),
        loss=getattr(keras.losses, compile_config['loss']['type'])(from_logits=compile_config['loss']['from_logits']),
        metrics=compile_config['metrics'],
    )
    return model

# Функция для обучения модели
def train_model(model, x_train, x_test, y_train, y_test, config):
    training_config = config['training']

    model_history = model.fit(x_train, y_train,
                            epochs=training_config['epochs'],
                            validation_data=(x_test, y_test),
                            batch_size=training_config['batch_size'],
                            callbacks=[
                                tf.keras.callbacks.ReduceLROnPlateau(monitor=training_config['monitor'],
                                                                    factor=training_config['factor'],
                                                                    patience=training_config['patience'],
                                                                    min_lr=training_config['min_lr']),
                            ])
    return model_history