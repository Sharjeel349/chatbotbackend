import json
import unittest

from FYP2.services.voice_service import (
    build_prompt,
    contains_urdu_script,
    direct_database_reply,
    encode_stream_event,
    response_language,
)


class VoiceHelperTests(unittest.TestCase):
    def test_urdu_voice_selects_roman_urdu(self):
        self.assertEqual(response_language("ur", "میرا سی جی پی اے"), "roman_urdu")

    def test_english_voice_stays_english(self):
        self.assertEqual(response_language("en", "What is my CGPA?"), "english")

    def test_roman_urdu_prompt_forbids_urdu_script(self):
        prompt = build_prompt(
            "میرا سی جی پی اے کیا ہے؟",
            "roman_urdu",
            "Neutral",
            {"profile": {"cgpa": 3.2}},
            "",
        )
        self.assertIn("Roman Urdu", prompt)
        self.assertIn("Never use Urdu or Arabic script", prompt)

    def test_direct_cgpa_answer_does_not_need_model(self):
        answer = direct_database_reply(
            "mera CGPA kya hai?",
            "roman_urdu",
            {"profile": {"cgpa": 3.2}},
        )
        self.assertEqual(answer, "Aap ka CGPA 3.2 hai.")
        self.assertFalse(contains_urdu_script(answer))

    def test_urdu_script_cgpa_question_also_skips_model(self):
        answer = direct_database_reply(
            "میرا سی جی پی اے کیا ہے؟",
            "roman_urdu",
            {"profile": {"cgpa": 3.2}},
        )
        self.assertEqual(answer, "Aap ka CGPA 3.2 hai.")

    def test_stream_event_is_one_json_line(self):
        raw = encode_stream_event({"type": "token", "text": "Aap "})
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw), {"type": "token", "text": "Aap "})


if __name__ == "__main__":
    unittest.main()
