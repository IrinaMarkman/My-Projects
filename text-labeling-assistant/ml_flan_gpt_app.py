import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Функция для FLAN (пример, замените на вашу реализацию)
from local_flant5 import get_local_label

def get_gpt_label(
    text: str,
    system_prompt: str = "Определи тональность текста: positive, negative или neutral.",
    api_key: str = None
) -> str:
    """
    Отправляет текст в GPT-3.5 и возвращает ответ.
    Требуется передать api_key, если он не установлен в окружении.
    """
    try:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return "Ошибка: API ключ не найден."

        client = OpenAI(api_key=key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка: {e}"

# Функция для ML модели
def load_ml_model():
    try:
        model = joblib.load("sentiment_classifier.pkl")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке ML модели: {e}")
        return None

st.set_page_config(page_title="Ассистент по разметке текста", layout="centered")
st.title("🧠 Ассистент по разметке текста")

option = st.radio(
    "Выбери метод разметки:",
    ("FLAN (бинарная классификация)", "ML модель", "GPT-3.5 via OpenAI API (экспериментально)")
)

# ==== FLAN ====
if option == "FLAN (бинарная классификация)":
    user_input = st.text_area("Введи текст для анализа", height=200)
    if st.button("Получить разметку"):
        if user_input.strip() == "":
            st.warning("Пожалуйста, введи текст.")
        else:
            with st.spinner("Модель размечает..."):
                try:
                    label = get_local_label(user_input)
                    st.success(f"Метка от FLAN модели: **{label}**")
                except Exception as e:
                    st.error(f"Ошибка при разметке: {e}")

    st.markdown("---")
    st.subheader("📄 Пакетная разметка для FLAN")

    uploaded_file_flan = st.file_uploader("Загрузи CSV с колонкой 'text'", type=["csv"], key="flan_label_csv")

    if uploaded_file_flan:
        try:
            df_flan = pd.read_csv(uploaded_file_flan)
            if 'text' not in df_flan.columns:
                st.error("CSV должен содержать колонку 'text'.")
            else:
                st.write("Файл загружен:")
                st.dataframe(df_flan.head())

                if st.button("Разметить текст из файла (FLAN)"):
                    with st.spinner("Пакетная разметка..."):
                        df_flan["predicted_label"] = df_flan["text"].apply(get_local_label)
                        st.success("Пакетная разметка завершена!")
                        st.dataframe(df_flan.head())

                        csv = df_flan.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Скачать CSV", data=csv, file_name="flan_labeled_data.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

# ==== ML модель ====
# ==== ML модель ====
elif option == "ML модель":
    st.subheader("📊 Оценка модели")

    uploaded_file_eval = st.file_uploader(
        "Загрузи CSV с колонками 'text' и 'label' для оценки", type=["csv"], key="ml_eval_csv"
    )

    if uploaded_file_eval:
        try:
            df_eval = pd.read_csv(uploaded_file_eval)
            if not {'text', 'label'}.issubset(df_eval.columns):
                st.error("CSV должен содержать колонки 'text' и 'label'.")
            else:
                st.write("Файл загружен для оценки:")
                st.dataframe(df_eval.head())

                model = load_ml_model()
                if model is None:
                    st.stop()

                if st.button("Оценить модель"):
                    with st.spinner("Оцениваем..."):
                        y_true = df_eval["label"]
                        y_pred = model.predict(df_eval["text"])
                        report = classification_report(y_true, y_pred, output_dict=True)

                        st.subheader("Отчёт по классификации")
                        st.text(classification_report(y_true, y_pred))

                        df_report = pd.DataFrame(report).transpose()
                        st.dataframe(df_report)

        except Exception as e:
            st.error(f"Ошибка при оценке: {e}")

    st.markdown("---")
    st.subheader("📄 Пакетная разметка")

    uploaded_file_predict = st.file_uploader(
        "Загрузи CSV с колонкой 'text' для разметки", type=["csv"], key="ml_predict_csv"
    )

    if uploaded_file_predict:
        try:
            df_pred = pd.read_csv(uploaded_file_predict)
            if 'text' not in df_pred.columns:
                st.error("CSV должен содержать колонку 'text'.")
            else:
                st.write("Файл загружен для разметки:")
                st.dataframe(df_pred.head())

                model = load_ml_model()
                if model is None:
                    st.stop()

                if st.button("Разметить текст (ML модель)"):
                    with st.spinner("Разметка..."):
                        df_pred["predicted_label"] = model.predict(df_pred["text"])
                        st.success("Разметка завершена!")
                        st.dataframe(df_pred.head())

                        csv = df_pred.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Скачать размеченный CSV",
                            data=csv,
                            file_name="ml_labeled_data.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Ошибка при разметке: {e}")
    
# ==== GPT-3.5 API ====
elif option == "GPT-3.5 via OpenAI API (экспериментально)":
    st.warning("⚠️ Требуется API-ключ OpenAI. Можно установить в переменных окружения или ввести здесь.")

    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key:
        st.info("API ключ взят из переменных окружения.")

    user_api_key = st.text_input("Введи свой OpenAI API ключ (если не установлен в окружении):", type="password", value="" if env_api_key else "")

    user_input = st.text_area("Введи текст для анализа", height=200)

    api_key_to_use = user_api_key if user_api_key.strip() else env_api_key

    if st.button("Получить разметку от GPT"):
        if not api_key_to_use or not user_input.strip():
            st.error("Пожалуйста, укажи API-ключ и текст.")
        else:
            try:
                with st.spinner("GPT размечает..."):
                    label = get_gpt_label(user_input, api_key=api_key_to_use)
                    st.success(f"Метка от GPT: **{label}**")
            except Exception as e:
                st.error(f"Ошибка при обращении к GPT: {e}")

    st.markdown("---")
    st.subheader("📄 Пакетная разметка через GPT")

    uploaded_file_gpt = st.file_uploader("Загрузи CSV с колонкой 'text'", type=["csv"], key="gpt_csv")

    if uploaded_file_gpt:
        try:
            df_gpt = pd.read_csv(uploaded_file_gpt)
            if 'text' not in df_gpt.columns:
                st.error("CSV должен содержать колонку 'text'.")
            else:
                st.write("Файл загружен:")
                st.dataframe(df_gpt.head())

                if st.button("Разметить текст через GPT"):
                    if not api_key_to_use:
                        st.error("API-ключ обязателен.")
                    else:
                        with st.spinner("Разметка через GPT..."):
                            df_gpt["predicted_label"] = df_gpt["text"].apply(
                                lambda x: get_gpt_label(x, api_key=api_key_to_use)
                            )
                            st.success("Разметка завершена!")
                            st.dataframe(df_gpt.head())

                            csv = df_gpt.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Скачать CSV", data=csv, file_name="gpt_labeled_data.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Ошибка при пакетной разметке через GPT: {e}")