import psycopg2
from psycopg2.extras import RealDictCursor

# Update these credentials to match your local setup
DB_CONFIG = {
    "host": "localhost",
    "database": "fyp_i",         # Make sure this matches your DB name
    "user": "Sharjeel",         # Your username
    "password": "Shamra349",    # Your password
    "port": 5432
}

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise e