import streamlit as st
import pandas as pd
import requests

# Заголовок для Streamlit
st.title("Flight Dashboard")
st.write("This dashboard shows information about flight departures and arrivals.")

# Получаем данные от FastAPI
response = requests.get("http://127.0.0.1:8001/flights")
data = response.json()  # Получаем данные в формате JSON

# Преобразуем в DataFrame для отображения
df = pd.DataFrame(data)

# Отображаем DataFrame
st.write("Flight Data:", df)

# Расчет метрик
total_flights = len(df)
earliest_departure = df['Departure'].min()
latest_arrival = df['Arrival'].max()

# Вывод метрик
st.write(f"Total number of flights: {total_flights}")
st.write(f"Earliest departure time: {earliest_departure}")
st.write(f"Latest arrival time: {latest_arrival}")

# Визуализация данных
st.write("Flight departure and arrival times:")
st.line_chart(df[['Departure', 'Arrival']])
