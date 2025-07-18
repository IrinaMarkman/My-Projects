# Анализ запросов Яндекс.Картинок: мобильные vs десктопные платформы

## О проекте
В этом проекте проводится анализ запросов пользователей Яндекс.Картинок с целью выявления различий в интересах и поведении пользователей мобильных устройств и компьютеров. Используются методы кластеризации, подсчёта распределений, временного анализа и визуализации.

## Файлы в репозитории
- `analysis.ipynb` — Jupyter-ноутбук с полным кодом анализа и визуализацией.
- `requirements.txt` — список необходимых Python-библиотек для запуска ноутбука.

## Запуск проекта
1. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Запустите Jupyter Notebook:
```bash
jupyter notebook image_search_analysis.ipynb
```
Следуйте инструкциям в ноутбуке для воспроизведения анализа.
