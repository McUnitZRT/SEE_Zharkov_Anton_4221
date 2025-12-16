import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Классификатор Фрукты vs Овощи",
    page_icon="🍎",
    layout="wide"
)


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/fruits_vegetables_model.h5')
        st.success("✅ Модель успешно загружена!")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None


def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image):
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image, verbose=0)[0][0]

    if prediction > 0.5:
        class_name = "🥦 ОВОЩ"
        confidence = prediction
        probability = prediction
    else:
        class_name = "🍎 ФРУКТ"
        confidence = 1 - prediction
        probability = prediction

    return class_name, confidence, probability


def plot_probabilities(probability):
    fig, ax = plt.subplots(figsize=(8, 2))

    classes = ['Фрукт', 'Овощ']
    probabilities = [1 - probability, probability]
    colors = ['#51cf66', '#ff6b6b']

    bars = ax.barh(classes, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Вероятность')
    ax.set_title('Распределение вероятностей')

    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{prob:.3f}', ha='left', va='center')

    return fig


def main():

    st.title("🍎 Классификатор Фрукты vs Овощи 🥦")
    st.markdown("---")

    with st.sidebar:
        # Статус модели
        st.header("🔧 Статус модели")
        try:
            model = load_model()
            if model is not None:
                st.success("✅ Модель готова")
            else:
                st.error("❌ Модель не загружена")
        except:
            st.error("❌ Модель не найдена")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Загрузка изображения")

        uploaded_file = st.file_uploader(
            "Выберите изображение...",
            type=['jpg', 'jpeg', 'png'],
            help="Поддерживаемые форматы: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
        else:
            image = None
            st.info("👆 Загрузите изображение фрукта или овоща")

            # Примеры
            with st.expander("📸 Примеры для тестирования"):
                st.write("""
                **Попробуйте найти изображения:**
                - 🍎 **Фрукты:** яблоко, банан, апельсин, клубника
                - 🥦 **Овощи:** морковь, брокколи, огурец, перец
                - 🍅 **Спорные:** помидор (ботанически - фрукт!)
                """)

    with col2:
        st.subheader("🔍 Результат классификации")

        if image is not None:
            model = load_model()

            if model is not None:
                with st.spinner("🔍 Анализируем изображение..."):
                    time.sleep(0.5)

                    class_name, confidence, probability = predict_image(model, image)

                st.markdown("### Результат:")

                if confidence > 0.8:
                    st.success(f"# {class_name}")
                    st.balloons()
                elif confidence > 0.6:
                    st.warning(f"# {class_name}")
                else:
                    st.error(f"# {class_name}")

                col_metric1, col_metric2 = st.columns(2)

                with col_metric1:
                    st.metric(
                        label="Уверенность модели",
                        value=f"{confidence * 100:.1f}%"
                    )

                with col_metric2:
                    fruit_prob = (1 - probability) * 100
                    veg_prob = probability * 100
                    st.metric(
                        label="Вероятность овоща",
                        value=f"{veg_prob:.1f}%"
                    )

                st.progress(float(confidence))
                st.pyplot(plot_probabilities(probability))

                with st.expander("📊 Детали предсказания"):
                    st.write(f"**Определенный класс:** {class_name}")
                    st.write(f"**Уверенность модели:** {confidence:.3f}")
                    st.write(f"**Сырое значение предсказания:** {probability:.3f}")
                    st.write(f"**Вероятность фрукта:** {probability * 100:.2f}%")
                    st.write(f"**Вероятность овоща:** {(1 - probability) * 100:.2f}%")

            else:
                st.error("Модель не загружена. Сначала обучите модель!")

        else:
            st.info("Здесь появится результат классификации")
            st.pyplot(plot_probabilities(0.5))


# Запуск приложения
if __name__ == "__main__":
    main()