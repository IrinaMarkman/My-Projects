import pytest
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from unittest.mock import patch
from dashboard_med.med_dash_sql_project import create_db, insert_data, execute_queries, create_pivot_table, visualize_data, fill_pivot_table, long_pivot, dash_create


@pytest.fixture
def setup_db():
    conn, cursor = create_db()
    yield conn, cursor
    conn.close()

def test_create_db(setup_db):
    conn, cursor = setup_db
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('patients', 'doctors');")
    tables = [row[0] for row in cursor.fetchall()]
    assert 'patients' in tables
    assert 'doctors' in tables
    cursor.execute("PRAGMA table_info(patients);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "patient_id" in columns

def test_insert_data(setup_db):
    conn, cursor = setup_db
    # Данные для users (patient_id, name, diagnosis, gender, age)
    patients_data = [(1, "Mary", "gastritis", "female",25)]
    # Данные для doctors (patient_id, clinic, doctor, price, order_date)
    doctors_data = [(1, "rosemed", "therapist", 2500, "2023-01-05")]
    # Вставка данных
    insert_data(conn, cursor, patients_data, doctors_data)
    # Проверка users
    cursor.execute("SELECT COUNT(*) FROM patients;")
    patients_count = cursor.fetchone()[0]
    assert patients_count == 1
    # Проверка orders
    cursor.execute("SELECT COUNT(*) FROM doctors;")
    doctors_count = cursor.fetchone()[0]
    assert doctors_count == 1

def test_execute_queries(setup_db):
    conn, cursor = setup_db
    # Данные для users (patient_id, name, diagnosis, gender, age)
    patients_data = [(1, "Mary", "gastritis", "female",25)]
    # Данные для doctors (patient_id, clinic, doctor, price, order_date)
    doctors_data = [(1, "rosemed", "therapist", 2500, "2023-01-05")]
    # Вставка данных
    insert_data(conn, cursor, patients_data, doctors_data)
    conn.commit()

    results = execute_queries(cursor)

    assert "Общее количество пациентов" in results
    assert results["Общее количество пациентов"].iloc[0, 0] == 1
    assert "Общая сумма за приемы" in results
    assert results["Общая сумма за приемы"].iloc[0, 0] == 2500

def test_create_pivot_table(setup_db):
    conn, cursor = setup_db

    patients_data = [
        (1, "Mary", "gastritis", "female", 25),
        (2, "John", "asthma", "male", 18)
    ]
    doctors_data = [
        (1, "rosemed", "therapist", 2500, "2023-01-05"),
        (2, "starmed", "therapist", 3000, "2024-02-07")
    ]

    insert_data(conn, cursor, patients_data, doctors_data)
    conn.commit()

    pivot_table = create_pivot_table(conn)

    # Убедимся, что есть 2 месяца (колонки)
    assert pivot_table.shape[1] == 2, "Должно быть 2 разных месяца в колонках"
    assert "2023-01" in pivot_table.columns
    assert "2024-02" in pivot_table.columns
    

def test_visualize_data(monkeypatch):
    import matplotlib.pyplot as plt

    # Заглушка, чтобы окно графика не открывалось
    monkeypatch.setattr(plt, "show", lambda: None)

    # Подготовка сводной таблицы
    data = {
        "2023-01": [2500],
        "2023-02": [7000],
        "2024-02": [5000]
    }
    index = ["rosemed"]
    pivot_table = pd.DataFrame(data, index=index)

    try:
        visualize_data(pivot_table)
    except Exception as e:
        pytest.fail(f"visualize_data выбросила исключение: {e}")

def test_visualize_data_calls_heatmap(monkeypatch):
    # Заглушаем plt.show()
    monkeypatch.setattr(plt, "show", lambda: None)

    data = {
        "2023-01": [2500],
        "2023-02": [7000],
        "2024-02": [5000]
    }
    pivot_table = pd.DataFrame(data, index=["rosemed"])

    with patch("seaborn.heatmap") as mock_heatmap:
        visualize_data(pivot_table)
        mock_heatmap.assert_called_once()

def test_fill_pivot_table(min_value=500, max_value=10000):
    # Определяем тестовые данные
    data = {
        "2023-01": [2500],
        "2023-02": [7000],
        "2024-02": [5000],
        "2025-03": [np.nan]  # Значение NaN для тестирования заполнения
    }
    
    # Создаем DataFrame
    pivot_table = pd.DataFrame(data, index=["rosemed"])

    # Применяем функцию для заполнения NaN значений случайными числами между min_value и max_value
    pivot_table_filled = fill_pivot_table(pivot_table)
    # Проверяем, что в DataFrame нет NaN значений
    assert pivot_table_filled.isnull().sum().sum() == 0  # Убедимся, что после заполнения нет NaN

def test_long_pivot(min_value=500, max_value=10000):
    data = {
        "2023-01": [2500],
        "2023-02": [7000],
        "2024-02": [5000],
        "2025-03": [np.nan]
    }

    pivot_table = pd.DataFrame(data, index=["rosemed"])
    pivot_table["clinic"] = pivot_table.index
    pivot_table.reset_index(drop=True, inplace=True)

    pivot_table_filled = pivot_table.apply(
        lambda x: x.apply(lambda val: np.random.randint(min_value, max_value) if pd.isna(val) else val)
    )

    assert pivot_table_filled.shape == (1, 5)  # 4 месяца + clinic

    df_long = long_pivot(pivot_table_filled)  # или передавай min_value, если нужно

    assert not df_long.empty
    assert "clinic" in df_long.columns

@patch("dash.Dash.run")  # Мокаем метод run
def test_dash_create(mock_run):
    data = {
        "2023-01": [2500, 3000],
        "2023-02": [7000, 5000],
        "2024-02": [5000, 7000]
    }

    pivot_table = pd.DataFrame(data, index=["rosemed", "bestmed"])
    pivot_table["clinic"] = pivot_table.index
    pivot_table.reset_index(drop=True, inplace=True)

    try:
        dash_create(pivot_table)
    except Exception as e:
        pytest.fail(f"dash_create выбросила исключение: {e}")

    # Проверка, что мок был вызван (app.run запустился, но "внутри" ничего не делал)
    mock_run.assert_called_once()




    


