import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def create_data_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        'dataset/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator


def create_model():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_model(model, validation_generator):
    """Оценка модели на валидационных данных"""
    print("\n" + "=" * 50)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 50)

    # Предсказания
    predictions = model.predict(validation_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = validation_generator.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Фрукты', 'Овощи']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Фрукты', 'Овощи'],
                yticklabels=['Фрукты', 'Овощи'])
    plt.title('Confusion Matrix')
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Точность
    accuracy = np.mean(y_pred == y_true)
    print(f"\nТочность на валидационных данных: {accuracy:.4f}")

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("Создание папок...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('dataset/train/fruits', exist_ok=True)
    os.makedirs('dataset/train/vegetables', exist_ok=True)

    print("Загрузка и подготовка данных...")
    train_generator, validation_generator = create_data_generators()

    if train_generator.samples == 0:
        print("ОШИБКА: В папке dataset/train нет изображений!")
        print("Пожалуйста, добавьте изображения в:")
        print("- dataset/train/fruits/")
        print("- dataset/train/vegetables/")
        return

    print(f"Классы: {train_generator.class_indices}")
    print(f"Тренировочные samples: {train_generator.samples}")
    print(f"Валидационные samples: {validation_generator.samples}")

    print("Создание модели...")
    model = create_model()

    print("Архитектура модели:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/fruits_vegetables_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    print("Начало обучения...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    print("Оценка модели...")
    evaluate_model(model, validation_generator)

    print("Визуализация результатов...")
    plot_training_history(history)

    print("Сохранение модели...")
    model.save('models/fruits_vegetables_model.h5')
    print("Модель успешно сохранена в models/fruits_vegetables_model.h5")


if __name__ == "__main__":
    main()