import os
import sqlite3
import pandas as pd
import numpy as np
import json
import base64
import io
from flask import Flask, render_template, request, g, redirect, url_for
import matplotlib.pyplot as plt
import shutil

app = Flask(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
DATABASE = os.path.join(BASE_DIR, 'forecast.db')
CSV_PATH = os.path.join(BASE_DIR, 'portfolio_data.csv')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- DB Helpers ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(_=None):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        filename TEXT,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        user TEXT,
        forecast_days INTEGER,
        forecast_result TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(dataset_id) REFERENCES datasets(id)
    )''')
    db.commit()

# --- Import portfolio_data.csv into a SQL table ---
def import_portfolio_to_sql():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        with sqlite3.connect(DATABASE) as conn:
            df.to_sql('portfolio', conn, if_exists='replace', index=False)
        print("[OK] portfolio_data.csv loaded into 'portfolio' SQL table.")

# --- Seed default dataset into uploads + tracking table ---
def ensure_default_dataset():
    db = get_db()
    cur = db.execute('SELECT COUNT(*) FROM datasets')
    if cur.fetchone()[0] == 0:
        dst = os.path.join(UPLOAD_FOLDER, 'portfolio_data.csv')
        if not os.path.exists(dst) and os.path.exists(CSV_PATH):
            shutil.copy(CSV_PATH, dst)
        db.execute('INSERT INTO datasets (name, filename) VALUES (?, ?)', ('Default Portfolio', 'portfolio_data.csv'))
        db.commit()

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    db = get_db()
    if request.method == 'POST':
        file = request.files.get('file')
        name = request.form.get('name') or (file.filename if file else None)
        if file and file.filename.endswith('.csv'):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Avoid overwriting files
            counter = 1
            base, ext = os.path.splitext(filename)
            while os.path.exists(filepath):
                filename = f"{base}_{counter}{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                counter += 1
            file.save(filepath)
            db.execute('INSERT INTO datasets (name, filename) VALUES (?, ?)', (name, filename))
            db.commit()
            return redirect(url_for('index'))
    cur = db.execute('SELECT id, name, filename, uploaded_at FROM datasets ORDER BY uploaded_at DESC')
    datasets = cur.fetchall()
    return render_template('index.html', datasets=datasets)

@app.route('/select/<int:dataset_id>')
def select_dataset(dataset_id):
    db = get_db()
    cur = db.execute('SELECT id, name, filename FROM datasets WHERE id=?', (dataset_id,))
    dataset = cur.fetchone()
    if not dataset:
        return 'Dataset not found', 404

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset['filename'])
    if not os.path.exists(filepath):
        return 'Dataset file not found', 404

    df = pd.read_csv(filepath)
    if 'Date' not in df.columns:
        return "CSV must contain a 'Date' column.", 400

    columns = [col for col in df.columns if col != 'Date']
    return render_template(
        'forecast.html',
        dataset=dataset,
        columns=columns,
        preview=df.head(10).to_html(index=False)
    )


@app.route('/forecast', methods=['POST'])
def forecast():
    user = request.form.get('user', 'Anonymous')
    dataset_id = int(request.form.get('dataset_id'))
    forecast_days = int(request.form.get('forecast_days', 7))
    db = get_db()

    # Get dataset info
    cur = db.execute('SELECT id, name, filename FROM datasets WHERE id=?', (dataset_id,))
    row = cur.fetchone()
    if not row:
        return "Dataset not found", 404

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], row['filename'])
    if not os.path.exists(filepath):
        return "Dataset file not found", 404

    df = pd.read_csv(filepath)  # ✅ Load the CSV here

    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    # Find the date column (case-insensitive)
    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
    if not date_col:
        return "CSV must have a 'Date' column.", 400

    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)

    # Check numeric columns
    value_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not value_columns:
        return "CSV must contain numeric columns for forecasting.", 400

    # SAFELY get last valid date
    valid_dates = df[date_col].dropna()
    if valid_dates.empty:
        return "No valid dates found in the 'Date' column.", 400

    last_date = valid_dates.max()
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)

    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    # Initialize forecast dictionary
    forecast = {col: [] for col in value_columns}

    # Generate simple forecasts (random walk for demo purposes)
    for col in value_columns:
        last_val = df[col].iloc[-1]
        for date in forecast_dates:
            last_val += np.random.randn()  # Random walk
            forecast[col].append({'date': date.strftime('%Y-%m-%d'), 'value': round(last_val, 2)})

    # Plot historical and forecasted data
    plt.figure(figsize=(10, 6))
    for col in value_columns:
        plt.plot(df[date_col], df[col], label=f'{col} History')
        plt.plot([pd.to_datetime(f['date']) for f in forecast[col]],
                 [f['value'] for f in forecast[col]],
                 label=f'{col} Forecast',
                 linestyle='--', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Forecast for {row["name"]}')
    plt.legend()
    plt.tight_layout()

    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Save forecast record
    db.execute('INSERT INTO forecasts (dataset_id, user, forecast_days, forecast_result) VALUES (?, ?, ?, ?)',
               (dataset_id, user, forecast_days, json.dumps(forecast)))
    db.commit()

    # Return the forecast.html page
    return render_template(
        'forecast.html',
        forecast=forecast,
        plot_image=plot_image,
        user=user,
        dataset_id=dataset_id,
        forecast_days=forecast_days,
        dataset=row,
        preview=df.head(10).to_html(index=False)
    )




@app.route('/history')
def history():
    db = get_db()
    cur = db.execute('''SELECT forecasts.id, datasets.name, forecasts.user, forecasts.forecast_days, forecasts.forecast_result, forecasts.timestamp
                        FROM forecasts JOIN datasets ON forecasts.dataset_id = datasets.id
                        ORDER BY forecasts.timestamp DESC''')
    logs_raw = cur.fetchall()

    logs = []
    for row in logs_raw:
        forecast_json = json.loads(row['forecast_result']) if row['forecast_result'] else []
        logs.append({
            'dataset': row['name'],
            'user': row['user'],
            'days': row['forecast_days'],
            'timestamp': row['timestamp'],
            'forecast': forecast_json
        })

    return render_template('history.html', logs=logs)


# --- Main Entry Point ---
if __name__ == '__main__':
    with app.app_context():
        init_db()
        ensure_default_dataset()
        import_portfolio_to_sql()
    app.run(debug=True)
