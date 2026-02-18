import sqlite3
from datetime import datetime

DB_NAME = "parking_logs.db"

def init_db():
    """Create the database table if it doesn't exist"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # יצירת טבלה: מזהה, מספר רכב, זמן, נתיב לתמונה, האם מורשה
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate_number TEXT,
                  timestamp DATETIME,
                  image_path TEXT,
                  is_authorized BOOLEAN,
                  confidence REAL)''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def log_entry(plate_number, image_path, is_authorized, confidence):
    """Save a new entry to the database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO entries (plate_number, timestamp, image_path, is_authorized, confidence) VALUES (?, ?, ?, ?, ?)",
              (plate_number, current_time, image_path, is_authorized, confidence))
    
    conn.commit()
    conn.close()
    print(f"Saved to DB: {plate_number}")

def get_all_entries():
    """Fetch all data for the dashboard"""
    conn = sqlite3.connect(DB_NAME)
    import pandas as pd
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# Initialize immediately when imported
init_db()