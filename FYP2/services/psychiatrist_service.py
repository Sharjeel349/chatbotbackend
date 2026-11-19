from __future__ import annotations

from collections import Counter
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_stress_state(state_raw: Optional[str]) -> str:
    s = str(state_raw or "").strip().upper()
    if not s:
        return "Neutral"
    if "SEVERE" in s and "STRESS" in s:
        return "Severe Stress"
    if "MODERATE" in s and "STRESS" in s:
        return "Moderate Stress"
    if "MILD" in s and "STRESS" in s:
        return "Mild Stress"
    if "STRESS" in s:
        return "Severe Stress"
    if "CALM" in s:
        return "Calm"
    if "FOCUS" in s or "ATTENT" in s:
        return "Focused"
    return "Neutral"


def _stress_severity_rank(state_label: str) -> int:
    normalized = _normalize_stress_state(state_label)
    if normalized == "Severe Stress":
        return 3
    if normalized == "Moderate Stress":
        return 2
    if normalized == "Mild Stress":
        return 1
    return 0


def _format_time_ago(ts: Optional[datetime]) -> str:
    if not ts:
        return "Recently"
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    diff = now - ts
    total_seconds = max(0, int(diff.total_seconds()))
    if diff.days <= 0:
        mins = total_seconds // 60
        if mins < 1:
            return "Just now"
        if mins < 60:
            return f"{mins} mins ago"
        return f"{mins // 60} hrs ago"
    if diff.days == 1:
        return "Yesterday"
    return f"{diff.days} days ago"


def get_stressed_students_dashboard(db_conn) -> Dict[str, Any]:
    """
    Return ONLY students who have at least one recorded stress event in either
    eeg_state or user-sent chat_message rows, aggregated with severity counts.
    """
    cur = db_conn.cursor()
    try:
        # 1. Fetch all stress events per student across eeg_state and user chat_message
        cur.execute(
            """
            SELECT
                ev.student_id,
                ev.source,
                ev.state,
                ev.ts
            FROM (
                SELECT
                    cs.student_id AS student_id,
                    'eeg' AS source,
                    es.mind_state AS state,
                    es.timestamp AS ts
                FROM eeg_state es
                JOIN chat_session cs ON es.session_id = cs.session_id
                WHERE es.mind_state IS NOT NULL
                  AND UPPER(es.mind_state) LIKE '%%STRESS%%'

                UNION ALL

                SELECT
                    cs.student_id AS student_id,
                    'chat' AS source,
                    cm.emotion AS state,
                    cm.timestamp AS ts
                FROM chat_message cm
                JOIN chat_session cs ON cm.session_id = cs.session_id
                WHERE cm.sender = 'User'
                  AND cm.emotion IS NOT NULL
                  AND UPPER(cm.emotion) LIKE '%%STRESS%%'
            ) ev
            ORDER BY ev.ts DESC NULLS LAST;
            """
        )
        stress_events = cur.fetchall()

        if not stress_events:
            return {
                "total_stressed_students": 0,
                "severe_stress_students": 0,
                "moderate_stress_students": 0,
                "mild_stress_students": 0,
                "total_stress_events": 0,
                "students": [],
            }

        # Aggregate stress stats per student_id
        stats_by_student: Dict[str, Dict[str, Any]] = {}
        for row in stress_events:
            sid = row["student_id"]
            if not sid:
                continue
            state_norm = _normalize_stress_state(row["state"])
            rank = _stress_severity_rank(state_norm)
            ts = row.get("ts")

            entry = stats_by_student.get(sid)
            if not entry:
                entry = {
                    "eeg_stress_count": 0,
                    "chat_stress_count": 0,
                    "peak_rank": rank,
                    "peak_stress_level": state_norm,
                    "latest_stress_state": state_norm,
                    "latest_stress_at": ts,
                }
                stats_by_student[sid] = entry

            if row["source"] == "eeg":
                entry["eeg_stress_count"] += 1
            else:
                entry["chat_stress_count"] += 1

            if rank > entry["peak_rank"]:
                entry["peak_rank"] = rank
                entry["peak_stress_level"] = state_norm

            if ts and (entry["latest_stress_at"] is None or ts > entry["latest_stress_at"]):
                entry["latest_stress_at"] = ts
                entry["latest_stress_state"] = state_norm

        stressed_sids = list(stats_by_student.keys())

        # 2. Fetch student profile details for stressed students only
        cur.execute(
            """
            SELECT
                s.sid AS student_id,
                u.name,
                u.email,
                COALESCE(cs.semester, 'N/A') AS semester,
                COALESCE(cs.section_code, 'N/A') AS section_code,
                s.cgpa
            FROM student s
            JOIN users u ON s.user_id = u.user_id
            LEFT JOIN class_section cs ON s.section_id = cs.section_id
            WHERE s.sid = ANY(%s);
            """,
            (stressed_sids,),
        )
        student_rows = cur.fetchall()

        # 3. Count EEG sessions per stressed student
        cur.execute(
            """
            SELECT cs.student_id, COUNT(DISTINCT cs.session_id) AS eeg_sessions_count
            FROM chat_session cs
            WHERE cs.student_id = ANY(%s)
              AND (
                  EXISTS (SELECT 1 FROM eeg_live_data eld WHERE eld.session_id = cs.session_id)
                  OR EXISTS (SELECT 1 FROM eeg_state es WHERE es.session_id = cs.session_id)
              )
            GROUP BY cs.student_id;
            """,
            (stressed_sids,),
        )
        session_counts = {
            r["student_id"]: int(r["eeg_sessions_count"] or 0)
            for r in cur.fetchall()
        }

        students: List[Dict[str, Any]] = []
        severe_students = 0
        moderate_students = 0
        mild_students = 0
        total_events = 0

        for sr in student_rows:
            sid = sr["student_id"]
            st = stats_by_student[sid]
            total_stu_events = st["eeg_stress_count"] + st["chat_stress_count"]
            total_events += total_stu_events

            peak = st["peak_stress_level"]
            if peak == "Severe Stress":
                severe_students += 1
            elif peak == "Moderate Stress":
                moderate_students += 1
            elif peak == "Mild Stress":
                mild_students += 1

            students.append(
                {
                    "student_id": sid,
                    "name": sr["name"],
                    "email": sr.get("email"),
                    "semester": str(sr.get("semester") or "N/A"),
                    "section_code": str(sr.get("section_code") or "N/A"),
                    "cgpa": float(sr["cgpa"]) if sr.get("cgpa") is not None else None,
                    "stress_events_count": total_stu_events,
                    "eeg_stress_count": st["eeg_stress_count"],
                    "chat_stress_count": st["chat_stress_count"],
                    "peak_stress_level": peak,
                    "latest_stress_state": st["latest_stress_state"],
                    "latest_stress_at": st["latest_stress_at"],
                    "latest_stress_time_ago": _format_time_ago(st["latest_stress_at"]),
                    "eeg_sessions_count": session_counts.get(sid, 0),
                }
            )

        # Sort by peak severity descending, then latest stress timestamp descending
        students.sort(
            key=lambda item: (
                _stress_severity_rank(item["peak_stress_level"]),
                item["latest_stress_at"] or datetime.min,
            ),
            reverse=True,
        )

        return {
            "total_stressed_students": len(students),
            "severe_stress_students": severe_students,
            "moderate_stress_students": moderate_students,
            "mild_stress_students": mild_students,
            "total_stress_events": total_events,
            "students": students,
        }
    finally:
        cur.close()


def _downsample_points(points: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    """Evenly downsample EEG points while preserving first, last, and peak-ratio points."""
    n = len(points)
    if n <= max_points or max_points < 3:
        return points

    peak_idx = max(range(n), key=lambda i: float(points[i].get("ratio") or 0.0))
    step = (n - 1) / float(max_points - 1)
    indices = {0, n - 1, peak_idx}
    for i in range(max_points):
        indices.add(int(round(i * step)))

    ordered = sorted(indices)
    return [points[idx] for idx in ordered]


def get_student_eeg_sessions(
    student_id: str,
    db_conn,
    max_points_per_session: int = 180,
) -> List[Dict[str, Any]]:
    """
    Fetch all EEG sessions for a student with continuous eeg_live_data points
    and 30-second eeg_state epoch summaries for plotting graphs.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT cs.session_id, COALESCE(cs.title, 'EEG Session') AS title, cs.start_time
            FROM chat_session cs
            WHERE cs.student_id = %s
              AND (
                  EXISTS (SELECT 1 FROM eeg_live_data eld WHERE eld.session_id = cs.session_id)
                  OR EXISTS (SELECT 1 FROM eeg_state es WHERE es.session_id = cs.session_id)
              )
            ORDER BY cs.start_time DESC NULLS LAST, cs.session_id DESC;
            """,
            (student_id,),
        )
        sessions = cur.fetchall()
        results: List[Dict[str, Any]] = []

        for sess in sessions:
            sid = sess["session_id"]

            # Continuous live points for graphing
            cur.execute(
                """
                SELECT eeg_data_id, alpha_power, beta_power, theta_power, ratio, mind_state, timestamp
                FROM eeg_live_data
                WHERE session_id = %s
                ORDER BY timestamp ASC, eeg_data_id ASC;
                """,
                (sid,),
            )
            live_rows = cur.fetchall()

            # 30-second aggregated epoch states
            cur.execute(
                """
                SELECT eeg_state_id, mind_state, timestamp
                FROM eeg_state
                WHERE session_id = %s
                ORDER BY timestamp ASC, eeg_state_id ASC;
                """,
                (sid,),
            )
            state_rows = cur.fetchall()

            epoch_states = [
                {
                    "eeg_state_id": int(sr["eeg_state_id"]),
                    "mind_state": _normalize_stress_state(sr.get("mind_state")),
                    "timestamp": sr.get("timestamp"),
                }
                for sr in state_rows
            ]

            total_points = len(live_rows)
            stress_points = 0
            ratios: List[float] = []
            state_counter: Counter = Counter()

            first_ts = live_rows[0]["timestamp"] if live_rows and live_rows[0].get("timestamp") else None
            formatted_points: List[Dict[str, Any]] = []

            for idx, r in enumerate(live_rows):
                ratio_val = float(r.get("ratio") or 0.0)
                st_norm = _normalize_stress_state(r.get("mind_state"))
                ratios.append(ratio_val)
                state_counter[st_norm] += 1
                if _stress_severity_rank(st_norm) > 0:
                    stress_points += 1

                ts = r.get("timestamp")
                if first_ts and ts:
                    elapsed = max(0.0, (ts - first_ts).total_seconds())
                else:
                    elapsed = float(idx)

                formatted_points.append(
                    {
                        "eeg_data_id": int(r["eeg_data_id"]),
                        "timestamp": ts,
                        "elapsed_seconds": round(elapsed, 2),
                        "alpha_power": round(float(r.get("alpha_power") or 0.0), 4),
                        "beta_power": round(float(r.get("beta_power") or 0.0), 4),
                        "theta_power": round(float(r.get("theta_power") or 0.0), 4),
                        "ratio": round(ratio_val, 4),
                        "mind_state": st_norm,
                    }
                )

            # Fallback if session only has eeg_state rows (no eeg_live_data)
            if not formatted_points and epoch_states:
                synthetic_ratio_map = {
                    "Calm": 0.55,
                    "Neutral": 0.95,
                    "Focused": 1.35,
                    "Mild Stress": 2.0,
                    "Moderate Stress": 2.7,
                    "Severe Stress": 3.3,
                }
                first_es_ts = epoch_states[0]["timestamp"]
                for idx, es in enumerate(epoch_states):
                    st_norm = es["mind_state"]
                    ratio_val = synthetic_ratio_map.get(st_norm, 1.0)
                    ratios.append(ratio_val)
                    state_counter[st_norm] += 1
                    if _stress_severity_rank(st_norm) > 0:
                        stress_points += 1
                    ts = es["timestamp"]
                    elapsed = max(0.0, (ts - first_es_ts).total_seconds()) if (first_es_ts and ts) else float(idx * 30)
                    formatted_points.append(
                        {
                            "eeg_data_id": es["eeg_state_id"],
                            "timestamp": ts,
                            "elapsed_seconds": round(elapsed, 2),
                            "alpha_power": 0.0,
                            "beta_power": 0.0,
                            "theta_power": 0.0,
                            "ratio": ratio_val,
                            "mind_state": st_norm,
                        }
                    )
                total_points = len(formatted_points)

            sampled_points = _downsample_points(formatted_points, max_points_per_session)
            avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else 0.0
            peak_ratio = round(max(ratios), 3) if ratios else 0.0
            dominant_state = state_counter.most_common(1)[0][0] if state_counter else "Neutral"

            results.append(
                {
                    "session_id": int(sid),
                    "title": sess["title"],
                    "start_time": sess.get("start_time"),
                    "total_points": total_points,
                    "stress_points": stress_points,
                    "avg_ratio": avg_ratio,
                    "peak_ratio": peak_ratio,
                    "dominant_state": dominant_state,
                    "data_points": sampled_points,
                    "epoch_states": epoch_states,
                }
            )

        return results
    finally:
        cur.close()


def get_student_stressed_messages(student_id: str, db_conn) -> List[Dict[str, Any]]:
    """
    Return ONLY student-sent messages flagged with a stress emotion.
    Note: To include bot responses or non-stressed session context later,
    relax the `cm.sender = 'User'` or `UPPER(cm.emotion) LIKE '%%STRESS%%'`
    clause in the SQL WHERE condition below.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT
                cm.message_id,
                cm.session_id,
                COALESCE(cs.title, 'Advisor Session') AS session_title,
                cm.sender,
                cm.message_text,
                cm.emotion,
                cm.timestamp
            FROM chat_message cm
            JOIN chat_session cs ON cm.session_id = cs.session_id
            WHERE cs.student_id = %s
              AND cm.sender = 'User'
              AND cm.emotion IS NOT NULL
              AND UPPER(cm.emotion) LIKE '%%STRESS%%'
            ORDER BY cm.timestamp DESC NULLS LAST;
            """,
            (student_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "message_id": int(r["message_id"]),
                "session_id": int(r["session_id"]),
                "session_title": r["session_title"],
                "sender": r["sender"],
                "message_text": r["message_text"] or "",
                "emotion": _normalize_stress_state(r.get("emotion")),
                "timestamp": r.get("timestamp"),
                "time_ago": _format_time_ago(r.get("timestamp")),
            }
            for r in rows
        ]
    finally:
        cur.close()
