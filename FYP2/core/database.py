import psycopg2
from psycopg2.extras import RealDictCursor

from FYP2.core.config import DB_CONFIG


def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    except psycopg2.Error as exc:
        raise ConnectionError("Could not connect to the academic database") from exc


def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
