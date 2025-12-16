# Распознование рукописных цифр
# Подключение библиотек:
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

# Загрузка данных из mnist:
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Нормализация данных
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Отображение первых 20 изображений из обучающей выборки
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.tight_layout()
plt.show()

# Создание модели
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),  # 1 скрытый слой
    Dense(64, activation='relu'),   # 2 скрытый слой
    Dense(10, activation='softmax')  # Выходной слой
])
# вывод структуры НС в консоль
print(model.summary())

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2, verbose=1)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nТочность на тестовых данных: {test_accuracy:.4f}")

# Тестирование
n = 5
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x, verbose=0)
print(f"\nПредсказание для изображения {n}:")
print(f"Вероятности: {res[0]}")
print(f"Предсказанная цифра: {np.argmax(res)}")
print(f"Истинная цифра: {y_test[n]}")

plt.figure(figsize=(6, 6))
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.title(f'Истинная цифра: {y_test[n]}, Предсказанная: {np.argmax(res)}')
plt.axis('off')
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test, verbose=0)
pred_classes = np.argmax(pred, axis=1)

print(f"\nФорма предсказаний: {pred_classes.shape}")

print("Первые 20 предсказаний:", pred_classes[:20])
print("Первые 20 истинных меток:", y_test[:20])

# Выделение неверных вариантов
mask = pred_classes == y_test
correct_count = np.sum(mask)
incorrect_count = len(mask) - correct_count

print(f"\nКоличество правильных предсказаний: {correct_count}/{len(y_test)}")
print(f"Количество ошибок: {incorrect_count}/{len(y_test)}")
print(f"Точность: {correct_count/len(y_test):.4f}")

x_false = x_test[~mask]
y_false_true = y_test[~mask]  # Истинные метки для ошибок
y_false_pred = pred_classes[~mask]  # Предсказанные метки для ошибок

print(f"\nФорма массива с ошибками: {x_false.shape}")

# Вывод первых 20 неверных результатов с подписями
plt.figure(figsize=(12, 12))
for i in range(min(20, len(x_false))):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.xlabel(f'True: {y_false_true[i]}, Pred: {y_false_pred[i]}',
               color='red' if y_false_true[i] != y_false_pred[i] else 'black')
plt.suptitle('Примеры ошибочных предсказаний', fontsize=16)
plt.tight_layout()
plt.show()


