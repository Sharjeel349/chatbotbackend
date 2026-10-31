import unittest
from unittest.mock import MagicMock, patch

from FYP2.services.voice_service import process_text_chat


class TextChatTests(unittest.TestCase):
    def test_empty_message_raises_error(self):
        mock_db = MagicMock()
        with self.assertRaises(ValueError):
            process_text_chat(session_id=1, message="   ", db_conn=mock_db)

    @patch("FYP2.services.voice_service.save_voice_messages")
    @patch("FYP2.services.voice_service.get_latest_emotion", return_value="CALM")
    def test_greeting_text_processing(self, mock_emotion, mock_save):
        mock_db = MagicMock()
        result = process_text_chat(
            session_id=99,
            message="Hello Advisor",
            db_conn=mock_db,
            student_id="2023-ARID-0274",
            emotion="CALM",
            generate_audio=False,
        )
        self.assertEqual(result["user_text"], "Hello Advisor")
        self.assertIn("academic advisor", result["bot_text"].lower())
        self.assertIsNone(result["user_audio_url"])
        self.assertIsNone(result["bot_audio_url"])
        self.assertEqual(result["user_emotion"], "CALM")
        self.assertEqual(result["response_language"], "english")
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()

