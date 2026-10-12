import unittest

from FYP2.services.advisor_graph import AdvisorState, advisor_graph
from FYP2.services.voice_service import (
    build_prompt,
    contains_urdu_script,
    encode_stream_event,
    response_language,
    retrieve_knowledge,
    select_tts_voice,
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

    def test_langgraph_greeting_intent(self):
        # English greeting
        state_en: AdvisorState = {
            "query": "Hi",
            "language": "english",
            "profile": {"name": "Sharjeel Ahmed"},
        }
        res_en = advisor_graph.invoke(state_en)
        self.assertEqual(res_en.get("intent"), "greeting")
        self.assertIn("Hello Sharjeel!", res_en.get("response", ""))
        self.assertIn("academic advisor", res_en.get("response", "").lower())

        # Roman Urdu greeting
        state_ur: AdvisorState = {
            "query": "Salam",
            "language": "roman_urdu",
            "profile": {"name": "Sharjeel Ahmed"},
        }
        res_ur = advisor_graph.invoke(state_ur)
        self.assertEqual(res_ur.get("intent"), "greeting")
        self.assertIn("Walekum Assalam Sharjeel!", res_ur.get("response", ""))

    def test_langgraph_irrelevant_query_guardrail(self):
        # Weather query in English
        state_weather: AdvisorState = {
            "query": "What's the weather today?",
            "language": "english",
        }
        res_weather = advisor_graph.invoke(state_weather)
        self.assertEqual(res_weather.get("intent"), "irrelevant")
        self.assertIn("academic advisor", res_weather.get("response", "").lower())
        self.assertIn("only assist with academic matters", res_weather.get("response", "").lower())

        # Joke / entertainment query
        state_joke: AdvisorState = {
            "query": "Tell me a funny joke",
            "language": "english",
        }
        res_joke = advisor_graph.invoke(state_joke)
        self.assertEqual(res_joke.get("intent"), "irrelevant")

        # Weather query in Roman Urdu
        state_weather_ur: AdvisorState = {
            "query": "aaj mosam kaisa hai?",
            "language": "roman_urdu",
        }
        res_weather_ur = advisor_graph.invoke(state_weather_ur)
        self.assertEqual(res_weather_ur.get("intent"), "irrelevant")
        self.assertIn("parhai", res_weather_ur.get("response", "").lower())

    def test_langgraph_academic_intent_and_knowledge_retrieval(self):
        # Failed courses query routes to academic and retrieves knowledge
        state_fail: AdvisorState = {
            "query": "what if i failed courses?",
            "language": "english",
            "profile": {"cgpa": 3.2},
        }
        retrieve_event = None
        for event in advisor_graph.stream(state_fail):
            if "retrieve_knowledge" in event:
                retrieve_event = event["retrieve_knowledge"]
                break

        self.assertIsNotNone(retrieve_event)
        knowledge = retrieve_event.get("knowledge", "")
        self.assertIn("Course Drop", knowledge)

        # CGPA query also routes to academic (no direct DB hijacking)
        state_cgpa: AdvisorState = {
            "query": "mera CGPA kya hai?",
            "language": "roman_urdu",
            "profile": {"cgpa": 3.2},
        }
        cgpa_event = None
        for event in advisor_graph.stream(state_cgpa):
            if "retrieve_knowledge" in event:
                cgpa_event = event["retrieve_knowledge"]
                break

        self.assertIsNotNone(cgpa_event)
        self.assertIn("prompt", cgpa_event)

    def test_tts_voice_selection_urdu_hindi(self):
        import pyttsx3

        engine = pyttsx3.init()
        # Roman Urdu should select Urdu or Hindi voice
        voice_roman_ur = select_tts_voice(engine, language="roman_urdu", detected_language="ur")
        if voice_roman_ur:
            self.assertTrue(
                any(k in voice_roman_ur.lower() for k in ["hi", "ur", "lekha", "hindi", "urdu", "rishi", "in"]),
                f"Expected Hindi/Urdu voice, got {voice_roman_ur}",
            )

        # English text when user spoke Urdu should also select Urdu or Hindi voice for correct pronunciation
        voice_en_with_ur = select_tts_voice(engine, language="english", detected_language="ur")
        if voice_en_with_ur:
            self.assertTrue(
                any(k in voice_en_with_ur.lower() for k in ["hi", "ur", "lekha", "hindi", "urdu", "rishi", "in"]),
                f"Expected Hindi/Urdu voice, got {voice_en_with_ur}",
            )

    def test_freeze_and_drop_policy_retrieval(self):
        k_freeze = retrieve_knowledge("Tell me about semester freeze policy in detail.")
        self.assertIn("Semester Freeze Policy", k_freeze)
        self.assertIn("2 semesters", k_freeze)

        k_for_ease = retrieve_knowledge("Can you tell me semester for ease, policy?")
        self.assertIn("Semester Freeze Policy", k_for_ease)

        k_drop = retrieve_knowledge("What is the course drop policy?")
        self.assertIn("Course Drop", k_drop)

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
