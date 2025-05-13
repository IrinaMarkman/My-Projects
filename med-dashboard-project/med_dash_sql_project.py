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

# 1. Создание базы данных
def create_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Удаляем таблицы, если они существуют
    cursor.execute('DROP TABLE IF EXISTS patients')
    cursor.execute('DROP TABLE IF EXISTS doctors')

    # Таблица пациентов
    cursor.execute('''
    CREATE TABLE patients (
        patient_id INTEGER PRIMARY KEY,
        name TEXT,
        diagnosis TEXT,
        gender TEXT,
        age INTEGER
    )
    ''')

    # Таблица докторов
    cursor.execute('''
    CREATE TABLE doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        clinic TEXT,
        doctor TEXT,
        price REAL,
        order_date TEXT
    )
    ''')

    print("Database created")
    return conn, cursor

# 2. Вставка данных
def insert_data(conn, cursor):
    cursor.execute('DELETE FROM patients')
    cursor.execute('DELETE FROM doctors')
    conn.commit()
    
    # --- Пациенты ---
    diagnoses = ['diabetes', 'gastritis', 'schizophrenia', 'hypertension', 'asthma']
    genders = ['female', 'male']
    patients_data = []

    for i in range(1, 21):  # 20 пациентов
        name = f"Patient {i}"
        diagnosis = random.choice(diagnoses)
        gender = random.choice(genders)
        age = random.randint(18, 80)
        patients_data.append((i, name, diagnosis, gender, age))

    cursor.executemany('''
        INSERT INTO patients (patient_id, name, diagnosis, gender, age)
        VALUES (?, ?, ?, ?, ?)
    ''', patients_data)

    # --- Визиты к врачам ---
    clinics = ['rosemed', 'starmed', 'bestmed']
    doctors = ['endocrinologist', 'gastroenterologist', 'psychiatrist', 'therapist', 'neurologist']
    base_prices = [1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200, 3500]

    start_date = datetime(2023, 1, 1)
    doctors_data = []

    for i in range(1, 101):  # 100 записей
        patient_id = random.randint(1, 20)
        clinic = random.choice(clinics)
        doctor = random.choice(doctors)
        price = round(random.choice(base_prices) + random.uniform(-150, 150), 2)
        order_date = (start_date + timedelta(days=random.randint(0, 480))).date().isoformat()

        doctors_data.append((patient_id, clinic, doctor, price, order_date))

    cursor.executemany('''
        INSERT INTO doctors (patient_id, clinic, doctor, price, order_date)
        VALUES (?, ?, ?, ?, ?)
    ''', doctors_data)

    conn.commit()

# 3. SQL-запросы
def execute_queries(cursor):
    queries = {
        "Общее количество пациентов": "SELECT COUNT(*) FROM patients",
        "Общая сумма за приемы": "SELECT ROUND(SUM(price)) AS revenue FROM doctors",
        "Количество уникальных диагнозов и клиник": '''
            SELECT COUNT(DISTINCT p.diagnosis) AS unique_diagnoses, COUNT(DISTINCT d.clinic) AS unique_clinics
            FROM patients p
            JOIN doctors d ON p.patient_id = d.patient_id
        ''',
        "Количество заказов на клинику": "SELECT clinic, COUNT(*) AS total_appointments FROM doctors GROUP BY clinic",
        "Прибыль клиники": "SELECT clinic, SUM(price) AS clinic_revenue FROM doctors GROUP BY clinic ORDER BY clinic_revenue DESC",
        "Прибыль по диагнозам": '''
            SELECT p.diagnosis, SUM(d.price) AS diagnosis_cost
            FROM patients p
            JOIN doctors d ON p.patient_id = d.patient_id
            GROUP BY diagnosis ORDER BY diagnosis_cost DESC
        '''
    }
    results = {}
    for title, q in queries.items():
        print(f"\n--- {title} ---")
        df = pd.read_sql_query(q, cursor.connection)
        print(df)
        results[title] = df
    return results

# 4. Создание сводной таблицы
def create_pivot_table(conn):
    query = '''
        SELECT
            clinic,
            strftime('%Y-%m', order_date) AS month,
            SUM(price) AS revenue
        FROM doctors
        GROUP BY clinic, month
        ORDER BY month, clinic;
    '''
    df = pd.read_sql_query(query, conn)

    # Получаем список всех уникальных клиник и месяцев
    clinics = df['clinic'].unique()
    months = df['month'].unique()

    # Создаём каркас всех возможных комбинаций клиника-месяц
    full_index = pd.MultiIndex.from_product([clinics, months], names=['clinic', 'month'])

    # Устанавливаем нужный индекс и переиндексируем
    df = df.set_index(['clinic', 'month']).reindex(full_index).reset_index()

    # Заполняем пропущенные значения выручки нулями
    df['revenue'] = df['revenue'].fillna(0)

    # Преобразуем в сводную таблицу
    pivot_table = df.pivot(index='clinic', columns='month', values='revenue')

    return pivot_table

# 5. Визуализация heatmap
def visualize_data(pivot_table):
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5, cbar_kws={'label': 'Доход'})
    plt.title("Прибыль по клиникам и месяцам")
    plt.xlabel("Месяц")
    plt.ylabel("Клиника")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def fill_pivot_table(pivot_table, min_value=500, max_value=10000):
    pivot_table_filled = pivot_table.apply(
        lambda x: x.apply(lambda val: np.random.randint(min_value, max_value) if pd.isna(val) else val)
    )
    return pivot_table_filled

# 7. Преобразование в длинный формат
def long_pivot(pivot_table_filled):
    return pivot_table_filled.reset_index().melt(id_vars="clinic", var_name="month", value_name="revenue")

# 8. Dash дашборд
def dash_create(pivot_table):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Преобразуем pivot_table в long формат для визуализации
    df_long = long_pivot(pivot_table)

    clinics = df_long['clinic'].unique().tolist()
    months = df_long['month'].unique().tolist()

    app.layout = dbc.Container([
        html.H2("Выручка по клиникам по месяцам"),
        dcc.Checklist(
            id='clinic-selector',
            options=[{'label': c, 'value': c} for c in clinics],
            value=clinics,  # по умолчанию все клиники отображаются
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        dcc.Graph(id='revenue-graph')
    ], fluid=True)  # Убедитесь, что аргумент `fluid` правильный

    @app.callback(
        Output('revenue-graph', 'figure'),
        Input('clinic-selector', 'value')
    )
    def update_graph(selected_clinics):
        fig = go.Figure()

        # Добавляем только те клиники, которые были выбраны
        for clinic in selected_clinics:
            clinic_data = df_long[df_long['clinic'] == clinic]
            fig.add_trace(
                go.Scatter(
                    x=clinic_data['month'],
                    y=clinic_data['revenue'],
                    mode='lines+markers',
                    name=clinic,
                    line=dict(shape='linear')
                )
            )

        fig.update_layout(
            title="Динамика выручки по клиникам",
            xaxis_title="Месяц",
            yaxis_title="Выручка",
            hovermode="x unified"
        )

        return fig

    app.run(debug=False)

if __name__ == '__main__':
    # Пример функции для обработки данных и запуска дашборда
    conn, cursor = create_db()
    insert_data(conn, cursor)
    execute_queries(cursor)
    pivot_table = create_pivot_table(conn)
    visualize_data(pivot_table)
    filled = fill_pivot_table(pivot_table)
    df_long = long_pivot(filled)

    # Запускаем создание дашборда
    dash_create(pivot_table)