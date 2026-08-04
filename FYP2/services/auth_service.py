import base64
import hashlib
import hmac
import logging
import os


logger = logging.getLogger(__name__)
PASSWORD_PREFIX = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 310_000
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return "$".join(
        (
            PASSWORD_PREFIX,
            str(iterations),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, stored_password: str) -> bool:
    if not stored_password.startswith(f"{PASSWORD_PREFIX}$"):
        # A successful legacy login is transparently upgraded below.
        return hmac.compare_digest(password, stored_password)

    try:
        _, iterations, salt, expected = stored_password.split("$", 3)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            base64.b64decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(
            base64.b64encode(digest).decode("ascii"),
            expected,
        )
    except (ValueError, TypeError):
        logger.warning("Stored password hash has an invalid format")
        return False


def authenticate_user(email, password, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT u.user_id, u.name, u.email, u.role, u.password,
                   s.sid AS student_id
            FROM Users u
            LEFT JOIN Student s ON u.user_id = s.user_id
            WHERE u.email = %s;
            """,
            (email,),
        )
        row = cur.fetchone()
        if not row or not verify_password(password, row["password"]):
            return None

        result = dict(row)
        stored_password = result.pop("password")
        if not stored_password.startswith(f"{PASSWORD_PREFIX}$"):
            try:
                cur.execute(
                    "UPDATE Users SET password = %s WHERE user_id = %s",
                    (hash_password(password), result["user_id"]),
                )
                db_conn.commit()
            except Exception:
                db_conn.rollback()
                logger.exception("Could not upgrade the legacy password hash")
        return result
    finally:
        cur.close()
