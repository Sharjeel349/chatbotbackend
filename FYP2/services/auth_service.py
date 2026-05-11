def authenticate_user(email, password, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT u.user_id, u.name, u.email, u.role, s.sid AS student_id
            FROM Users u
            LEFT JOIN Student s ON u.user_id = s.user_id
            WHERE u.email = %s AND u.password = %s;
        """
        cur.execute(query, (email, password))
        return cur.fetchone()
    finally:
        cur.close()