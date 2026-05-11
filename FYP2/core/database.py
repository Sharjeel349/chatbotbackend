import psycopg2
from psycopg2.extras import RealDictCursor
from FYP.core.config import DB_CONFIG

def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    except psycopg2.Error as e:
        print("DB Connection Error:", e)
        raise e

def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


