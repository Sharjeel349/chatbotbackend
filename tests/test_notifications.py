import unittest
from FYP2.core.database import get_connection
from FYP2.services import notification_service, teacher_service


class NotificationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = get_connection()
        cls.cur = cls.conn.cursor()
        # Clean test notifications
        cls.cur.execute("DELETE FROM student_notification WHERE title LIKE 'TEST_%';")
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM student_notification WHERE title LIKE 'TEST_%';")
        cls.conn.commit()
        cls.conn.close()

    def test_direct_and_broadcast_notifications(self):
        test_student_a = "2023-ARID-TEST-A"
        test_student_b = "2023-ARID-TEST-B"

        # 1. Create a meeting notification for student A
        notif_meeting = notification_service.create_notification(
            student_id=test_student_a,
            teacher_id=15,
            title="TEST_Meeting with Advisor",
            message="Meeting regarding course prerequisites",
            notification_type="meeting",
            meeting_time="Tomorrow 10:00 AM",
            location="Room 302",
            teacher_name="Dr. Advisor",
            db_conn=self.conn,
        )
        self.assertIsNotNone(notif_meeting.get("notification_id"))
        self.assertEqual(notif_meeting["student_id"], test_student_a)
        self.assertEqual(notif_meeting["notification_type"], "meeting")

        # 2. Create a broadcast notification (student_id = None)
        notif_broadcast = notification_service.create_notification(
            student_id=None,
            teacher_id=15,
            title="TEST_Campus Announcement",
            message="Midterm exams scheduled next week",
            notification_type="broadcast",
            db_conn=self.conn,
        )
        self.assertIsNotNone(notif_broadcast.get("notification_id"))
        self.assertIsNone(notif_broadcast["student_id"])

        # 3. Check unread count for Student A (should see at least meeting + broadcast = 2)
        count_a = notification_service.get_student_unread_count(test_student_a, self.conn)
        self.assertGreaterEqual(count_a, 2)

        # 4. Student B should only see broadcast (not Student A's meeting)
        notifs_b = notification_service.get_student_notifications(test_student_b, unread_only=True, db_conn=self.conn)
        b_ids = [n["notification_id"] for n in notifs_b]
        self.assertIn(notif_broadcast["notification_id"], b_ids)
        self.assertNotIn(notif_meeting["notification_id"], b_ids)

        # 5. Student A marks the meeting as read
        ok = notification_service.mark_notification_as_read(test_student_a, notif_meeting["notification_id"], self.conn)
        self.assertTrue(ok)

        # 6. Verify that meeting notification NO LONGER appears for Student A in unread list
        notifs_a = notification_service.get_student_notifications(test_student_a, unread_only=True, db_conn=self.conn)
        a_ids = [n["notification_id"] for n in notifs_a]
        self.assertNotIn(notif_meeting["notification_id"], a_ids)

        # 7. Mark all remaining read for Student A
        notification_service.mark_all_as_read(test_student_a, self.conn)
        final_count_a = notification_service.get_student_unread_count(test_student_a, self.conn)
        self.assertEqual(final_count_a, 0)

        # Student B should still have the broadcast unread
        count_b = notification_service.get_student_unread_count(test_student_b, self.conn)
        self.assertGreaterEqual(count_b, 1)


if __name__ == "__main__":
    unittest.main()
