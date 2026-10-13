import sqlite3
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(BASE_DIR / "fraudlens.db"))
)


def get_connection():
    
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)



def create_tables():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(amount, prediction, fraud_probability):
    connection = get_connection()
    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (amount, prediction, fraud_probability)
            VALUES (?, ?, ?)
        """, (amount, prediction, fraud_probability))

        connection.commit()
    finally:
        connection.close()


def get_predictions():
    connection = get_connection()
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, amount, prediction,
                   fraud_probability, created_at
            FROM predictions
            ORDER BY id DESC
        """)

        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
if __name__ == "__main__":
    create_tables()
    print("FraudLens database created successfully!")
