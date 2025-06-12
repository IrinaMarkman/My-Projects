import os
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

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