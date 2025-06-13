import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Загрузит переменные из .env

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY не установлен в переменных окружения")

client = OpenAI(api_key=api_key)

def get_summary_gpt(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты — полезный ассистент, который кратко резюмирует текст."},
                {"role": "user", "content": f"Сделай краткое содержание этого текста:\n\n{text}"}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        print(f"Ошибка при запросе к OpenAI: {e}")
        return "Ошибка при генерации краткого содержания"