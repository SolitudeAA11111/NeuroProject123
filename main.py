#main
import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
                             QFileDialog, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import tensorflow as tf
import keras
from PIL import Image
import yaml
from neuro import *

class NeuralNetworkGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_images = []
        self.database_path = ""
        self.yaml_path = ""
        self.model = None
        self.config = None

    def initUI(self):
        self.setWindowTitle('Нейронная сеть для классификации изображений')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        # Секция выбора изображений
        image_layout = QHBoxLayout()
        self.image_labels = []
        for i in range(3):
            label = QLabel(self)
            label.setFixedSize(150, 150)
            label.setAlignment(Qt.AlignCenter)
            label.setText(f"Изображение {i+1}")
            self.image_labels.append(label)
            image_layout.addWidget(label)
        layout.addLayout(image_layout)

        btn_select_images = QPushButton('Выбрать 3 изображения', self)
        btn_select_images.clicked.connect(self.select_images)
        layout.addWidget(btn_select_images)

        # Секция выбора базы данных
        self.db_label = QLabel('База данных: Не выбрана', self)
        layout.addWidget(self.db_label)
        btn_select_db = QPushButton('Выбрать базу данных', self)
        btn_select_db.clicked.connect(self.select_database)
        layout.addWidget(btn_select_db)

        # Секция выбора YAML файла
        self.yaml_label = QLabel('YAML файл: Не выбран', self)
        layout.addWidget(self.yaml_label)
        btn_select_yaml = QPushButton('Выбрать YAML файл', self)
        btn_select_yaml.clicked.connect(self.select_yaml)
        layout.addWidget(btn_select_yaml)

        # Секция выбора модели .keras
        self.model_label = QLabel('Модель: Не выбрана', self)
        layout.addWidget(self.model_label)
        btn_select_model = QPushButton('Выбрать модель .keras', self)
        btn_select_model.clicked.connect(self.select_model)
        layout.addWidget(btn_select_model)

        # Кнопки действий
        action_layout = QGridLayout()
        btn_train = QPushButton('Обучить модель', self)
        btn_train.clicked.connect(self.train_model)
        action_layout.addWidget(btn_train, 0, 0)

        btn_retrain = QPushButton('Переобучить модель', self)
        btn_retrain.clicked.connect(self.retrain_model)
        action_layout.addWidget(btn_retrain, 0, 1)

        btn_predict = QPushButton('Предсказание', self)
        btn_predict.clicked.connect(self.predict)
        action_layout.addWidget(btn_predict, 1, 0)

        layout.addLayout(action_layout)
        self.setLayout(layout)

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите 3 изображения", "",
                                                "Images (*.png *.jpg *.JPG *.jpeg)")
        if len(files) == 3:
            self.selected_images = files
            for i, file in enumerate(files):
                pixmap = QPixmap(file).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_labels[i].setPixmap(pixmap)

    def select_database(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите базу данных изображений")
        if folder:
            self.database_path = folder
            self.db_label.setText(f"База данных: {folder}")

    def select_yaml(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите YAML файл", "", "YAML Files (*.yaml)")
        if file:
            self.yaml_path = file
            self.yaml_label.setText(f"YAML файл: {file}")
            with open(file, 'r') as f:
                self.config = yaml.safe_load(f)
            print("YAML файл загружен успешно.")

    def select_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите модель .keras", "", "keras Files (*.keras)")
        if file:
            self.model_path = file
            self.model_label.setText(f"Модель: {file}")
            try:
                self.model = keras.saving.load_model(file)
                print("Модель загружена успешно.")
            except Exception as e:
                print(f"Ошибка при загрузке модели: {e}")

    def train_model(self):
            if not self.database_path or not self.yaml_path:
                print("Выберите базу данных и YAML файл перед обучением")
                return
            
            print("Обучение модели...")
            images, classes, _ = load_dataset(self.database_path, cache_file="preprocessed_data.npz")
            x_train, x_test, y_train, y_test = create_test_train_data(images, classes)
            
            model = create_model(len(classes), self.config)
            
            train_model(model, x_train, x_test, y_train, y_test, self.config)
            
            print("Обучение завершено")
            test_loss, test_accuracy = model.evaluate(x_test, y_test)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Точность на тестовых данных: {test_accuracy:.4f}")
            
            # Сохранение модели
            model_save_path = 'trained_model.keras'
            keras.saving.save_model(model, model_save_path)
            print(f"Модель сохранена в {model_save_path}")
            self.model = model

    def retrain_model(self):
        if not self.model or not (self.database_path and self.yaml_path):
            print("Сначала выберите базу данных и YAML файл для переобучения")
            return
        print("Переобучение модели...")
        
        # Загрузка новых данных
        images, classes, _ = load_dataset(self.database_path, cache_file="preprocessed_data.npz")
        x_train, x_test, y_train, y_test = create_test_train_data(images, classes)
        
        # Дообучение модели
        train_model(self.model, x_train, x_test, y_train, y_test, self.config)
        
        print("Переобучение завершено")
        
        # Оценка переобученной модели
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Точность на тестовых данных после переобучения: {test_accuracy:.4f}")

            # Сохранение переобученной модели
        model_save_path = 'retrained_model.keras'
        keras.saving.save_model(self.model, 'retrained_model.keras')
        print(f"Переобученная модель сохранена в {model_save_path}")

    def predict(self):
        if not self.model or not self.selected_images:
            print("Сначала обучите модель и выберите изображения")
            return

        def print_predictions(prediction, inverse_mapping, image_number):
            print(f"\nПредсказания для изображения {image_number}:")
            top_10_indices = np.argsort(prediction[0])[-10:][::-1]
            for j, idx in enumerate(top_10_indices, 1):
                class_name = inverse_mapping.get(idx, f"Неизвестный класс {idx}")
                probability = prediction[0][idx]
                print(f"{j}. {class_name}: {probability:.4f}")

            predicted_class = np.argmax(prediction)
            if predicted_class in inverse_mapping:
                predicted_old_class = inverse_mapping[predicted_class]
                print(f"Предсказанный класс для изображения {image_number}: {predicted_old_class}")
            else:
                print(f"Ошибка: класс {predicted_class} отсутствует в inverse_mapping.")


        print("Выполнение предсказания...")
        _, _, inverse_mapping = load_dataset(self.database_path, cache_file="preprocessed_data.npz")
        

        aggregated_predictions = 0
        for i, image_path in enumerate(self.selected_images, 1):
            img = Image.open(image_path)
            img = rgb_img_to_gray(img)
            img = np.expand_dims(img, axis=0)
            prediction = self.model.predict(img)
            
            print_predictions(prediction, inverse_mapping, i)
            
            if aggregated_predictions is None:
                aggregated_predictions = prediction
            else:
                aggregated_predictions += prediction

        aggregated_predictions /= len(self.selected_images)
        top_10_indices = np.argsort(aggregated_predictions[0])[-10:][::-1]
        print("Топ 10 предсказаний:")
        for i, idx in enumerate(top_10_indices, 1):
            class_name = inverse_mapping.get(idx, f"Неизвестный класс {idx}")
            probability = aggregated_predictions[0][idx]
            print(f"{i}. {class_name}: {probability:.4f}")

        predicted_new_class = np.argmax(aggregated_predictions)
        


        print(f"predicted_new_class: {predicted_new_class}")
        
        # Добавляем проверку на существование класса в inverse_mapping
        if predicted_new_class in inverse_mapping:
            predicted_old_class = inverse_mapping[predicted_new_class]
            print(f"Итоговый предсказанный класс для 3 изображений: {predicted_old_class}")
        else:
            print(f"Ошибка: класс {predicted_new_class} отсутствует в inverse_mapping.")
            print("Доступные классы в inverse_mapping:", list(inverse_mapping.keys()))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NeuralNetworkGUI()
    ex.show()
    sys.exit(app.exec_())