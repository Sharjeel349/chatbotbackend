import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def ensure_notifications_tables_exist(db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS student_notification (
                notification_id SERIAL PRIMARY KEY,
                student_id VARCHAR(50) NULL,
                teacher_id INT NULL,
                teacher_name VARCHAR(100) NULL,
                title VARCHAR(255) NOT NULL,
                message TEXT NOT NULL,
                notification_type VARCHAR(50) NOT NULL DEFAULT 'general',
                meeting_time VARCHAR(100) NULL,
                location VARCHAR(255) NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS student_notification_read (
                read_id SERIAL PRIMARY KEY,
                notification_id INT REFERENCES student_notification(notification_id) ON DELETE CASCADE,
                student_id VARCHAR(50) NOT NULL,
                read_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unq_notification_student UNIQUE(notification_id, student_id)
            );
            """
        )
        db_conn.commit()
    except Exception as e:
        logger.error(f"Error ensuring notification tables exist: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass
    finally:
        cur.close()


def create_notification(
    student_id: Optional[str],
    teacher_id: Optional[int],
    title: str,
    message: str,
    notification_type: str = "general",
    meeting_time: Optional[str] = None,
    location: Optional[str] = None,
    teacher_name: Optional[str] = None,
    db_conn = None,
) -> Dict[str, Any]:
    ensure_notifications_tables_exist(db_conn)
    cur = db_conn.cursor()
    try:
        # If teacher_name not provided, attempt to fetch from DB
        resolved_teacher_name = teacher_name
        if teacher_id and not resolved_teacher_name:
            cur.execute(
                """
                SELECT u.name
                FROM users u
                JOIN teacher t ON u.user_id = t.user_id
                WHERE t.teacher_id = %s;
                """,
                (teacher_id,),
            )
            row = cur.fetchone()
            if row and row.get("name"):
                resolved_teacher_name = row["name"]

        cur.execute(
            """
            INSERT INTO student_notification (
                student_id, teacher_id, teacher_name, title, message,
                notification_type, meeting_time, location
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING notification_id, student_id, teacher_id, teacher_name,
                      title, message, notification_type, meeting_time, location, created_at;
            """,
            (
                student_id if student_id else None,
                teacher_id,
                resolved_teacher_name,
                title,
                message,
                notification_type,
                meeting_time,
                location,
            ),
        )
        notif = cur.fetchone()
        db_conn.commit()
        notif["is_read"] = False
        return notif
    except Exception as e:
        logger.error(f"Failed to create notification: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass
        raise
    finally:
        cur.close()


def get_student_notifications(
    student_id: str,
    unread_only: bool = True,
    db_conn = None,
) -> List[Dict[str, Any]]:
    ensure_notifications_tables_exist(db_conn)
    cur = db_conn.cursor()
    try:
        if unread_only:
            cur.execute(
                """
                SELECT n.notification_id, n.student_id, n.teacher_id, n.teacher_name,
                       n.title, n.message, n.notification_type, n.meeting_time, n.location,
                       n.created_at, FALSE AS is_read
                FROM student_notification n
                LEFT JOIN student_notification_read r
                  ON n.notification_id = r.notification_id AND r.student_id = %s
                WHERE (n.student_id = %s OR n.student_id IS NULL)
                  AND r.read_id IS NULL
                ORDER BY n.created_at DESC;
                """,
                (student_id, student_id),
            )
        else:
            cur.execute(
                """
                SELECT n.notification_id, n.student_id, n.teacher_id, n.teacher_name,
                       n.title, n.message, n.notification_type, n.meeting_time, n.location,
                       n.created_at,
                       (CASE WHEN r.read_id IS NOT NULL THEN TRUE ELSE FALSE END) AS is_read
                FROM student_notification n
                LEFT JOIN student_notification_read r
                  ON n.notification_id = r.notification_id AND r.student_id = %s
                WHERE (n.student_id = %s OR n.student_id IS NULL)
                ORDER BY n.created_at DESC;
                """,
                (student_id, student_id),
            )
        rows = cur.fetchall()
        return rows
    except Exception as e:
        logger.error(f"Error fetching notifications for student {student_id}: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass
        return []
    finally:
        cur.close()


def get_student_unread_count(student_id: str, db_conn = None) -> int:
    ensure_notifications_tables_exist(db_conn)
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM student_notification n
            LEFT JOIN student_notification_read r
              ON n.notification_id = r.notification_id AND r.student_id = %s
            WHERE (n.student_id = %s OR n.student_id IS NULL)
              AND r.read_id IS NULL;
            """,
            (student_id, student_id),
        )
        row = cur.fetchone()
        return int(row["count"]) if row else 0
    except Exception as e:
        logger.error(f"Error fetching unread count for {student_id}: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        cur.close()


def mark_notification_as_read(student_id: str, notification_id: int, db_conn = None) -> bool:
    ensure_notifications_tables_exist(db_conn)
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO student_notification_read (notification_id, student_id)
            VALUES (%s, %s)
            ON CONFLICT (notification_id, student_id) DO NOTHING;
            """,
            (notification_id, student_id),
        )
        db_conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read for {student_id}: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass
        return False
    finally:
        cur.close()


def mark_all_as_read(student_id: str, db_conn = None) -> int:
    ensure_notifications_tables_exist(db_conn)
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO student_notification_read (notification_id, student_id)
            SELECT n.notification_id, %s
            FROM student_notification n
            LEFT JOIN student_notification_read r
              ON n.notification_id = r.notification_id AND r.student_id = %s
            WHERE (n.student_id = %s OR n.student_id IS NULL)
              AND r.read_id IS NULL
            ON CONFLICT (notification_id, student_id) DO NOTHING;
            """,
            (student_id, student_id, student_id),
        )
        affected = cur.rowcount
        db_conn.commit()
        return affected
    except Exception as e:
        logger.error(f"Error marking all notifications as read for {student_id}: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        cur.close()
