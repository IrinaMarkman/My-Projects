import os
import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH")
SPREADSHEET_ID = os.getenv("GOOGLE_SHEET_ID")

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def save_to_gsheet(df: pd.DataFrame) -> str | None:
    try:
        print("🟡 Загружаем ключ авторизации из файла:", SERVICE_ACCOUNT_FILE)
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        print("🟢 Создаём клиент Google Sheets")
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        values = [df.columns.to_list()] + df.values.tolist()
        body = {'values': values}

        print("📤 Пытаемся записать данные в таблицу...")
        result = sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range='Лист1!A1',  # <= Убедись, что это точное имя вкладки
            valueInputOption='RAW',
            body=body
        ).execute()

        print("✅ Успешно записано:", result)
        sheet_url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
        return sheet_url

    except Exception as e:
        print(f"❌ Ошибка при сохранении в Google Sheets: {e}")
        return None
