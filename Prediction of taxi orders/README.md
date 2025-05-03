# Прогнозирование количества заказов такси 🚖

## Задача 🎯
Компания "Чётенькое такси" хочет прогнозировать количество заказов такси на следующий час, чтобы улучшить планирование работы водителей в периоды пиковой нагрузки. Требуется построить модель с метрикой RMSE не более 48.

## Описание данных 📊
Данные представлены в файле `taxi.csv`:
- **datetime** — временная метка заказа.
- **num_orders** — количество заказов за указанный час.

## Подготовка данных 🧹
Данные были загружены и преобразованы, временной ряд ресемплирован по часам.

```python
df = pd.read_csv('taxi.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df = df.resample('1H').sum()
Модели 🏗️
Были протестированы следующие модели:

Ridge Regression 🤖

LGBMRegressor 📊

CatBoostRegressor 🍃

Гиперпараметры 🔧:
Ridge: alpha = [0.005, 0.02, 0.03], solver = ['auto', 'lsqr']

LGBMRegressor: n_estimators = [50, 100], max_depth = [10, 20]

CatBoostRegressor: iterations = [10, 15], depth = [7, 9]

Тестирование 🧪
После подбора гиперпараметров модели показали следующие результаты на валидации:

LGBMRegressor: RMSE = 24.93 ✅

Ridge Regression: RMSE = 27.07 ❌

CatBoostRegressor: RMSE = 25.95 🟡

Результаты 📈
Лучшая модель: LGBMRegressor (n_estimators=50, max_depth=10) 🏆

На тестовой выборке RMSE = 40.51, что удовлетворяет требованиям проекта (RMSE ≤ 48). ✔️

Используемые библиотеки 📚
pandas 🐼

matplotlib 📊

catboost 🍃

lightgbm 💡

statsmodels 📉

scikit-learn ⚙️


