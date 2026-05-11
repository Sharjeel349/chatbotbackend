def get_chat_sessions(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT session_id, title, start_time
            FROM Chat_Session
            WHERE student_id = %s
            ORDER BY start_time DESC;
        """
        cur.execute(query, (student_id,))
        return cur.fetchall()
    finally:
        cur.close()

def get_chat_messages(session_id: int, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT message_id, sender, message_text, timestamp , emotion,voice_file
            FROM Chat_Message
            WHERE session_id = %s
            ORDER BY timestamp ASC;
        """
        cur.execute(query, (session_id,))
        return cur.fetchall()
    finally:
        cur.close()


def create_chat_session(student_id: str, title: str, db_conn):
    cur = db_conn.cursor()
    try:
        # Insert the record and return the generated fields
        query = """
            INSERT INTO Chat_Session (student_id, title)
            VALUES (%s, %s)
            RETURNING session_id, title, start_time;
        """
        cur.execute(query, (student_id, title))
        new_session = cur.fetchone()

        # You must commit the transaction when inserting data!
        db_conn.commit()

        return new_session
    except Exception as e:
        db_conn.rollback()
        raise e
    finally:
        cur.close()


def save_chat_message(session_id: int, sender: str, message_text: str, voice_file: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            INSERT INTO Chat_Message (session_id, sender, message_text, voice_file)
            VALUES (%s, %s, %s, %s)
            RETURNING message_id, sender, message_text, timestamp;
        """
        cur.execute(query, (session_id, sender, message_text, voice_file))
        new_msg = cur.fetchone()
        db_conn.commit()

        if new_msg:
            return {
                "message_id": new_msg[0],
                "sender": new_msg[1],
                "message_text": new_msg[2],
                "timestamp": new_msg[3]
            }
        return None
    except Exception as e:
        db_conn.rollback()
        raise e
    finally:
        cur.close()
