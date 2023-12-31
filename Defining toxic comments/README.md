# Обучение модели классификации комментариев для интернет-магазина
[ipynb](https://github.com/IrinaMarkman/My-Projects/blob/main/Defining%20toxic%20comments/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%20%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8%20%D0%BA%D0%BE%D0%BC%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%80%D0%B8%D0%B5%D0%B2%20%D0%B4%D0%BB%D1%8F%20%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D0%BD%D0%B5%D1%82-%D0%BC%D0%B0%D0%B3%D0%B0%D0%B7%D0%B8%D0%BD%D0%B0.ipynb) 
## Описание проекта
Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. Нужно обучить модель классифицировать комментарии на позитивные и негативные. Имеется набор данных с разметкой о токсичности правок.

Метрики качества *F1* должна быть не меньше 0.75. 
## Навыки и инструменты
* python
* pandas
* numpy
* catboost
* re
* spacy
* sklearn.model_selection.RandomizedSearchCV
* sklearn.feature_extraction.text.TfidfVectorizer
* sklearn.compose.ColumnTransformer
* sklearn.pipeline.Pipeline
* sklearn.linear_model.LogisticRegression
* sklearn.metrics.f1_score
* sklearn.model_selection.GridSearchCV
## Общий вывод
Обучено две модели: LogisticRegression (в пайплайне с TfidfVectorizer())  и CatBoostClassifier.
* Параметры LogisticRegression - C=10, class_weight='balanced', random_state=42, TfidVectorizer - ngram_range: (1, 1). F1 score - 0.7553 на тестовых данных.
* f1 score у CatBoostClassifier без настройки гиперпараметров - 0.7557.
* Обе модели удовлетворяют условиям задачи, однако у CatBoostClassifier даже без настройки гиперпараметров метрика немного лучше.
