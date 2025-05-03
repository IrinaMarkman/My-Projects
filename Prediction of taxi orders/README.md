# Прогнозирование количества заказов такси

## Задача

Компания "Чётенькое такси" собирает данные о заказах такси и хочет оптимизировать работу с водителями, особенно в периоды пиковой нагрузки. Для этого нужно спрогнозировать количество заказов такси на следующий час. Задача заключалась в построении модели для предсказания количества заказов.

Метрика качества: RMSE (Root Mean Squared Error) не должна превышать 48.

## Описание данных

Данные находятся в файле `taxi.csv`. Столбцы:
- **datetime** — временная метка заказа.
- **num_orders** — количество заказов в этот момент времени.

## План работы

1. Загрузка и подготовка данных.
2. Проведение анализа данных.
3. Обучение различных моделей с настройкой гиперпараметров.
4. Оценка качества моделей на тестовой выборке.
5. Прогнозирование на основе лучших моделей.

## Подготовка данных

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Загрузка данных
df = pd.read_csv("taxi.csv")

# Преобразование столбца datetime в тип datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Ресемплирование данных по часам, используя сумму заказов за каждый час
df = df.resample('1H').sum()
Анализ данных
Проверены пропущенные значения — их нет.

Построены графики тренда и сезонности данных.

Выполнен анализ с помощью скользящего среднего и стандартного отклонения.

python
Копировать
Редактировать
# Тренд
decomposed = seasonal_decompose(df)
plt.figure(figsize=(8, 3))
decomposed.trend.plot(ax=plt.gca())
plt.title('Тренд')
plt.xlabel('Месяц')
plt.ylabel('Количество заказов')

# Сезонность
plt.figure(figsize=(8, 3))
decomposed.seasonal['2018-08-01':"2018-08-02"].plot(ax=plt.gca())
plt.title('Сезонность')
plt.xlabel('Время')
plt.ylabel('Количество заказов')
Обучение моделей
Для обучения использовались следующие модели:

Ridge Regression

LGBMRegressor

CatBoostRegressor

Гиперпараметры:
Ridge: alpha = [0.005, 0.02, 0.03, 0.05, 0.06], solver = ['auto', 'lsqr', 'cholesky']

LGBMRegressor: n_estimators = [50, 100, 150], max_depth = [10, 20]

CatBoostRegressor: iterations = [10, 15], depth = [7, 9]

Использованные инструменты:
RandomizedSearchCV для поиска лучших гиперпараметров.

TimeSeriesSplit для кросс-валидации временных рядов.

Тестирование
Модель LGBMRegressor показала лучший результат с метрикой RMSE 24.93 на обучающей выборке.

python
Копировать
Редактировать
model = LGBMRegressor(n_estimators=50, max_depth=10, random_state=12)
model.fit(train_features, train_target)

# Прогнозирование
pred = model.predict(test_features)

# Оценка качества модели
rmse = mean_squared_error(test_target, pred, squared=False)
print(rmse)
Результаты
Лучшие гиперпараметры для моделей:

LGBMRegressor: n_estimators=50, max_depth=10 → RMSE = 24.93

Ridge: solver='lsqr', alpha=0.06 → RMSE = 27.07

CatBoostRegressor: iterations=13, depth=7 → RMSE = 25.95

Лучшая модель: LGBMRegressor показала лучший результат с метрикой RMSE = 24.93 на тестовой выборке.

Тестирование на адекватность:

Прогнозирование на основе предыдущего часа (без модели): RMSE = 58.85.

Модель LGBMRegressor с RMSE = 40.51 показывает лучший результат.

Выводы
LGBMRegressor с гиперпараметрами n_estimators=50 и max_depth=10 — лучшая модель для прогнозирования количества заказов такси на следующий час.

Результаты модели соответствуют условиям задачи, RMSE на тестовой выборке составляет 40.5, что меньше требуемых 48.

Требования
Установленные библиотеки:
pandas

matplotlib

catboost

lightgbm

statsmodels

sklearn

seaborn

txt
Копировать
Редактировать
pandas==1.4.3
matplotlib==3.4.3
catboost==1.0.6
lightgbm==3.3.1
statsmodels==0.13.2
sklearn==0.24.2
seaborn==0.11.2
markdown
Копировать
Редактировать


