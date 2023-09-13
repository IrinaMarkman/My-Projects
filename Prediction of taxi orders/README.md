# Прогнозирование количества заказов такси на следующий час
[ipynb](https://github.com/IrinaMarkman/My-Projects/blob/main/Prediction%20of%20taxi%20orders/%D0%9F%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B7%D0%B0%D0%BA%D0%B0%D0%B7%D0%BE%D0%B2%20%D1%82%D0%B0%D0%BA%D1%81%D0%B8.ipynb) 
## Описание проекта
Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Нужно построить модель для такого предсказания. Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.
## Навыки и инструменты
* python
* pandas
* statsmodels
* matplotlib
* catboost
* lightgbm
* sklearn.linear_model.Ridge
* sklearn.model_selection.RandomizedSearchCV
* sklearn.metrics.mean_squared_error
* sklearn.model_selection.TimeSeriesSplit
## Общий вывод
Построена модель LGBMRegressor c гиперпараметрами max_depth = 10, n_estimators = 50, которая поможет спрогнозировать количество заказов такси на следующий час и оптимизировать работу компании по предоставлению такси. Метрика RMSE нашей модели на тестовой выборке 40.5.
