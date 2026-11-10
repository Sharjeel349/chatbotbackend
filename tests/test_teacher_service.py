import unittest
from unittest.mock import MagicMock
from FYP2.services import teacher_service


class TestTeacherService(unittest.TestCase):
    def test_normalize_cognitive_state(self):
        self.assertEqual(teacher_service._normalize_cognitive_state("CALM"), "Calm")
        self.assertEqual(teacher_service._normalize_cognitive_state("calm"), "Calm")
        self.assertEqual(teacher_service._normalize_cognitive_state("FOCUSED"), "Focused")
        self.assertEqual(teacher_service._normalize_cognitive_state("attentive"), "Focused")
        self.assertEqual(teacher_service._normalize_cognitive_state("MILD STRESS"), "Mild Stress")
        self.assertEqual(teacher_service._normalize_cognitive_state("moderate stress"), "Moderate Stress")
        self.assertEqual(teacher_service._normalize_cognitive_state("severe stress"), "Severe Stress")
        self.assertEqual(teacher_service._normalize_cognitive_state("STRESSED"), "Stressed")
        self.assertEqual(teacher_service._normalize_cognitive_state("Neutral"), "Neutral")
        self.assertEqual(teacher_service._normalize_cognitive_state(None), "Neutral")

    def test_get_semester_number(self):
        self.assertEqual(teacher_service._get_semester_number("6th"), 6)
        self.assertEqual(teacher_service._get_semester_number("Semester 4"), 4)
        self.assertEqual(teacher_service._get_semester_number("3"), 3)
        self.assertEqual(teacher_service._get_semester_number("None"), 1)

    def test_get_eeg_cognitive_analytics_mock(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        # First query (EEG_State counts)
        # Second query (chat_message emotion counts)
        # Third query (combined recent telemetry)
        mock_cur.fetchall.side_effect = [
            [{"state": "CALM", "count": 10}, {"state": "Neutral", "count": 10}],
            [{"state": "Focused", "count": 5}, {"state": "Mild Stress", "count": 5}],
            [
                {
                    "student_name": "Test Student",
                    "student_sid": "2023-ARID-0001",
                    "course": "Advising",
                    "timestamp": None,
                    "state": "Calm",
                }
            ],
        ]

        result = teacher_service.get_eeg_cognitive_analytics(1, mock_conn)
        self.assertEqual(result["total_evaluations"], 30)
        self.assertIn("Calm", result["mind_state_percentages"])
        self.assertIn("Focused", result["mind_state_percentages"])
        self.assertIn("Mild Stress", result["mind_state_percentages"])
        self.assertIn("Stressed", result["mind_state_percentages"])
        self.assertEqual(len(result["recent_eeg_records"]), 1)
        self.assertEqual(result["recent_eeg_records"][0]["state"], "Calm")
        self.assertTrue(len(result["insights"]) > 0)


if __name__ == "__main__":
    unittest.main()
