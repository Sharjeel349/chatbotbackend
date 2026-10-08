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

    def test_direct_greeting_reply(self):
        reply_en = direct_database_reply(
            "Hi",
            "english",
            {"profile": {"name": "Sharjeel Ahmed"}},
        )
        self.assertIn("Hello Sharjeel!", reply_en)
        self.assertIn("academic advisor", reply_en)

        reply_ur = direct_database_reply(
            "Salam",
            "roman_urdu",
            {"profile": {"name": "Sharjeel Ahmed"}},
        )
        self.assertIn("Walekum Assalam Sharjeel!", reply_ur)

    def test_robust_urdu_transliteration(self):
        from FYP2.services.roman_urdu_service import transliterate_urdu_to_roman
        urdu_voice_text = "میری اینوستی کی فیس ابھی تک سبب نہیں ہوئی کیا میں اوی بھی اگزان دیستکتا ہوں"
        roman = transliterate_urdu_to_roman(urdu_voice_text)
        self.assertIn("university", roman)
        self.assertIn("fees", roman)
        self.assertIn("submit", roman)
        self.assertIn("exam", roman)
        self.assertIn("de sakta hoon", roman)
        self.assertFalse(contains_urdu_script(roman))


if __name__ == "__main__":
    unittest.main()

