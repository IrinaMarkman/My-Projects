import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_db(db_name="example.db"):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        item TEXT,
        quantity INTEGER,
        price REAL,
        order_date TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    return conn, cursor

def insert_data(cursor, users_data=None, orders_data=None):
    if users_data is None:
        users_data = [
            ('John Doe', 'john.doe@example.com'),
            ('Jane Smith', 'jane.smith@example.com'),
            ('Tom Lee', 'tom.lee@example.com'),
            ('Alice Brown', 'alice.brown@example.com'),
            ('Charlie Black', 'charlie.black@example.com')
        ]

    cursor.executemany('''
        INSERT INTO users (name, email) VALUES (?, ?)
    ''', users_data)

    if orders_data is None:
        np.random.seed(42)
        items = ['Book', 'Laptop', 'Pen', 'Headphones', 'Monitor']
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        orders_data = [
            (np.random.randint(1, 6), np.random.choice(items),
             np.random.randint(1, 5), round(np.random.uniform(5, 500), 2),
             pd.to_datetime(np.random.choice(dates)).strftime('%Y-%m-%d'))
            for _ in range(100)
        ]

    cursor.executemany('''
        INSERT INTO orders (user_id, item, quantity, price, order_date)
        VALUES (?, ?, ?, ?, ?)
    ''', orders_data)

def execute_queries(conn):
    queries = {
        "Общее количество заказов": "SELECT COUNT(*) AS total_orders FROM orders",
        "Общий доход": "SELECT ROUND(SUM(quantity * price), 2) AS revenue FROM orders",
        "Средний чек": "SELECT ROUND(AVG(quantity * price), 2) AS avg_order FROM orders",
        "Самый популярный товар": """
            SELECT item, COUNT(*) AS freq 
            FROM orders 
            GROUP BY item 
            ORDER BY freq DESC 
            LIMIT 1
        """,
        "Доход по дням": """
            SELECT order_date, ROUND(SUM(quantity * price), 2) AS daily_revenue
            FROM orders 
            GROUP BY order_date 
            ORDER BY order_date
        """
    }

    results = {}
    for title, query in queries.items():
        df = pd.read_sql_query(query, conn)
        results[title] = df
        print(f"\n--- {title} ---")
        print(df)

    return results

def visualize_data(results):
    daily = results["Доход по дням"]
    daily['order_date'] = pd.to_datetime(daily['order_date'])
    plt.figure(figsize=(12, 5))
    plt.plot(daily['order_date'], daily['daily_revenue'], marker='o')
    plt.title('Доход по дням')
    plt.xlabel('Дата')
    plt.ylabel('Доход')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def close_db(conn):
    conn.close()
    
def full_pipeline():
    try:
        conn, cursor = create_db()
        cursor = insert_data(cursor)
        results = execute_queries(conn)
        visualize_data(results)
        d = dtale.show(results["Доход по дням"])
        d.open_browser()
    finally:
        close_db(conn)
