from fastapi import FastAPI
import pandas as pd
import numpy as np

# Генерация данных для рейсов
def generate_flight_data():
    np.random.seed(42)
    flight_ids = ['AA101', 'BB202', 'CC303', 'DD404', 'EE505']
    destinations = ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Miami']
    departure_times = pd.date_range('2023-01-01', periods=5, freq='12H')
    arrival_times = departure_times + pd.to_timedelta(np.random.randint(60, 180, size=5), unit='m')

    data = {
        'Flight': flight_ids,
        'Departure': departure_times,
        'Arrival': arrival_times,
        'Destination': destinations,
    }
    return pd.DataFrame(data)

app = FastAPI()

# Корневой маршрут
@app.get("/")
async def root():
    return {"message": "Welcome to the Flight Dashboard API!"}

# Эндпоинт для получения данных рейсов
@app.get("/flights")
async def get_flights():
    df = generate_flight_data()
    return df.to_dict(orient="records")

# Запуск FastAPI на порту 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
