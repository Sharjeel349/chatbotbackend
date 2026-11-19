import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from FYP2.services import psychiatrist_service


class TestPsychiatristService(unittest.TestCase):
    def test_normalize_stress_state(self):
        self.assertEqual(psychiatrist_service._normalize_stress_state("Mild Stress"), "Mild Stress")
        self.assertEqual(psychiatrist_service._normalize_stress_state("MODERATE STRESS"), "Moderate Stress")
        self.assertEqual(psychiatrist_service._normalize_stress_state("severe stress"), "Severe Stress")
        self.assertEqual(psychiatrist_service._normalize_stress_state("STRESSED"), "Severe Stress")
        self.assertEqual(psychiatrist_service._normalize_stress_state("Calm"), "Calm")
        self.assertEqual(psychiatrist_service._normalize_stress_state("Focused"), "Focused")
        self.assertEqual(psychiatrist_service._normalize_stress_state(None), "Neutral")

    def test_downsample_preserves_endpoints_and_peak(self):
        points = [{"eeg_data_id": i, "ratio": 1.0} for i in range(500)]
        points[137]["ratio"] = 9.5
        sampled = psychiatrist_service._downsample_points(points, max_points=50)
        self.assertLessEqual(len(sampled), 51)
        self.assertEqual(sampled[0]["eeg_data_id"], 0)
        self.assertEqual(sampled[-1]["eeg_data_id"], 499)
        self.assertTrue(any(p["eeg_data_id"] == 137 for p in sampled))

    def test_get_stressed_students_dashboard_only_returns_stressed(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        now = datetime.now()
        mock_cur.fetchall.side_effect = [
            [
                {"student_id": "2023-ARID-0001", "source": "eeg", "state": "Moderate Stress", "ts": now},
                {"student_id": "2023-ARID-0001", "source": "chat", "state": "Severe Stress", "ts": now - timedelta(minutes=5)},
                {"student_id": "2023-ARID-0002", "source": "eeg", "state": "Mild Stress", "ts": now - timedelta(hours=2)},
            ],
            [
                {
                    "student_id": "2023-ARID-0001",
                    "name": "Alice",
                    "email": "alice@biit.edu",
                    "semester": "6",
                    "section_code": "A",
                    "cgpa": 3.2,
                },
                {
                    "student_id": "2023-ARID-0002",
                    "name": "Bob",
                    "email": "bob@biit.edu",
                    "semester": "4",
                    "section_code": "B",
                    "cgpa": 2.9,
                },
            ],
            [
                {"student_id": "2023-ARID-0001", "eeg_sessions_count": 2},
                {"student_id": "2023-ARID-0002", "eeg_sessions_count": 1},
            ],
        ]

        res = psychiatrist_service.get_stressed_students_dashboard(mock_conn)
        self.assertEqual(res["total_stressed_students"], 2)
        self.assertEqual(res["severe_stress_students"], 1)
        self.assertEqual(res["mild_stress_students"], 1)
        self.assertEqual(res["total_stress_events"], 3)
        # Highest severity student first
        self.assertEqual(res["students"][0]["student_id"], "2023-ARID-0001")
        self.assertEqual(res["students"][0]["peak_stress_level"], "Severe Stress")
        self.assertEqual(res["students"][0]["eeg_stress_count"], 1)
        self.assertEqual(res["students"][0]["chat_stress_count"], 1)

    def test_get_student_stressed_messages_mock(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        mock_cur.fetchall.return_value = [
            {
                "message_id": 10,
                "session_id": 3,
                "session_title": "Midterm Worries",
                "sender": "User",
                "message_text": "I feel overwhelmed by my failed courses",
                "emotion": "Moderate Stress",
                "timestamp": datetime.now(),
            }
        ]

        msgs = psychiatrist_service.get_student_stressed_messages("2023-ARID-0001", mock_conn)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["sender"], "User")
        self.assertEqual(msgs[0]["emotion"], "Moderate Stress")


if __name__ == "__main__":
    unittest.main()
