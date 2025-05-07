import os
import sqlite3
import pytest
from pipeline.pipeline import create_db, insert_data, execute_queries, visualize_data, close_db, full_pipeline
import matplotlib.pyplot as plt
import pandas as pd

TEST_DB = "test.db"

@pytest.fixture
def setup_db():
    conn, cursor = create_db(TEST_DB)
    yield conn, cursor
    close_db(conn)
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

def test_create_db(setup_db):
    conn, cursor = setup_db
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('users', 'orders');")
    tables = [row[0] for row in cursor.fetchall()]
    assert 'users' in tables
    assert 'orders' in tables

def test_insert_data(setup_db):
    conn, cursor = setup_db
    # Данные для users
    users_data = [("Test User", "test@example.com")]
    # Данные для orders (например: user_id, product, amount)
    orders_data = [(1, "Laptop", 1, 1500.0, "2024-01-01")]
    # Вставка данных
    insert_data(cursor, users_data, orders_data)
    # Проверка users
    cursor.execute("SELECT COUNT(*) FROM users;")
    users_count = cursor.fetchone()[0]
    assert users_count == 1
    # Проверка orders
    cursor.execute("SELECT COUNT(*) FROM orders;")
    orders_count = cursor.fetchone()[0]
    assert orders_count == 1

def test_execute_queries(setup_db):
    conn, cursor = setup_db
    users_data = [("Test User", "test@example.com")]
    orders_data = [(1, "Test Item", 2, 100.0, "2024-01-01")]
    
    insert_data(cursor, users_data, orders_data)
    conn.commit()

    results = execute_queries(conn)

    assert "Общее количество заказов" in results
    assert results["Общее количество заказов"].iloc[0, 0] == 1
        
def test_visualize_data(monkeypatch):
    # Заглушка для plt.show(), чтобы не открывалось окно графика
    monkeypatch.setattr(plt, "show", lambda: None)
    # Подготовка фиктивных данных
    data = {
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "daily_revenue": [100.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)
    results = {"Доход по дням": df}
    # Вызов функции, исключений быть не должно
    try:
        visualize_data(results)
    except Exception as e:
        pytest.fail(f"visualize_data выбросила исключение: {e}")

def test_close_db():
    conn = sqlite3.connect(":memory:")
    close_db(conn)    
    # Проверяем, что соединение действительно закрыто
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # Операция после закрытия должна вызвать ошибку
