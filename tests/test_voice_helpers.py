import unittest

from FYP2.services.advisor_graph import AdvisorState, advisor_graph
from FYP2.services.voice_service import (
    build_prompt,
    contains_urdu_script,
    encode_stream_event,
    response_language,
    retrieve_knowledge,
    select_tts_voice,
    strip_markdown_for_tts,
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

    def test_langgraph_identity_intent(self):
        # English identity inquiry
        state_arid_en: AdvisorState = {
            "query": "U don't know my arid number??",
            "language": "english",
            "student_id": "2023-ARID-0274",
            "profile": {"name": "SHARJEEL AHMED", "registration_no": "2023-ARID-0274", "semester": "8", "cgpa": 3.86},
        }
        res_arid_en = advisor_graph.invoke(state_arid_en)
        self.assertEqual(res_arid_en.get("intent"), "identity")
        self.assertIn("2023-ARID-0274", res_arid_en.get("response", ""))
        self.assertIn("SHARJEEL", res_arid_en.get("response", ""))

        # Roman Urdu identity inquiry
        state_arid_ur: AdvisorState = {
            "query": "mera arid number kya hai?",
            "language": "roman_urdu",
            "student_id": "2023-ARID-0274",
            "profile": {"name": "SHARJEEL AHMED", "registration_no": "2023-ARID-0274", "semester": "8", "cgpa": 3.86},
        }
        res_arid_ur = advisor_graph.invoke(state_arid_ur)
        self.assertEqual(res_arid_ur.get("intent"), "identity")
        self.assertIn("2023-ARID-0274", res_arid_ur.get("response", ""))

        # Stressed student identity inquiry receives reassuring response
        state_arid_stressed: AdvisorState = {
            "query": "U don't know my arid number??",
            "language": "english",
            "emotion": "Stressed",
            "student_id": "2023-ARID-0274",
            "profile": {"name": "SHARJEEL AHMED", "registration_no": "2023-ARID-0274", "semester": "8", "cgpa": 3.86},
        }
        res_arid_stressed = advisor_graph.invoke(state_arid_stressed)
        self.assertIn("2023-ARID-0274", res_arid_stressed.get("response", ""))
        self.assertTrue(
            any(w in res_arid_stressed.get("response", "").lower() for w in ["deep breath", "don't worry", "support", "wonderfully"]),
            f"Expected supportive phrasing, got {res_arid_stressed.get('response')}"
        )

    def test_prompt_includes_stress_support_when_student_stressed(self):
        prompt_stressed = build_prompt(
            "I want to know my academic history",
            "english",
            "Stressed",
            {
                "profile": {
                    "name": "SHARJEEL AHMED",
                    "registration_no": "2023-ARID-0274",
                    "semester": "8",
                    "section_code": "A",
                    "cgpa": 3.86,
                }
            },
            "",
        )
        self.assertIn("STUDENT IS CURRENTLY STRESSED", prompt_stressed)
        self.assertIn("comforting and reassuring", prompt_stressed)
        self.assertIn("MEDIUM DETAIL", prompt_stressed)

    def test_prompt_includes_authenticated_identity_and_directives(self):
        prompt = build_prompt(
            "I want to know my academic history",
            "english",
            "Neutral",
            {
                "profile": {
                    "name": "SHARJEEL AHMED",
                    "registration_no": "2023-ARID-0274",
                    "semester": "8",
                    "section_code": "A",
                    "cgpa": 3.86,
                }
            },
            "",
        )
        self.assertIn("2023-ARID-0274", prompt)
        self.assertIn("SHARJEEL AHMED", prompt)
        self.assertIn("Never ask the student for their ARID number", prompt)
        self.assertIn("logged in to the portal", prompt)

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

        # Policy query also routes to academic and retrieves knowledge
        state_policy: AdvisorState = {
            "query": "what is the semester freeze policy?",
            "language": "english",
            "profile": {"cgpa": 3.2},
        }
        policy_event = None
        for event in advisor_graph.stream(state_policy):
            if "retrieve_knowledge" in event:
                policy_event = event["retrieve_knowledge"]
                break

        self.assertIsNotNone(policy_event)
        self.assertIn("prompt", policy_event)

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

    def test_degree_completion_zero_failed_courses_stressed(self):
        state: AdvisorState = {
            "query": "kya mery degree complete ho jai gy",
            "language": "roman_urdu",
            "emotion": "STRESSED",
            "student_id": "2023-ARID-0274",
            "profile": {
                "name": "SHARJEEL AHMED",
                "registration_no": "2023-ARID-0274",
                "cgpa": 3.86,
                "semester": "8",
                "section_code": "A",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "degree_completion")
        reply = res.get("response", "")
        self.assertIn("Pareshan mat hon", reply)
        self.assertIn("complete ho jaye gi", reply)
        self.assertIn("fail nahi hai", reply)
        self.assertIn("3.86", reply)
        self.assertIn("12 semesters", reply)
        # Must NEVER tell student with 0 failed courses to repeat courses
        self.assertNotIn("repeat karna", reply)
        self.assertNotIn("prerequisite course ko first pass", reply)

    def test_degree_completion_zero_failed_courses_neutral(self):
        state: AdvisorState = {
            "query": "kya mery degree complete ho jai gy",
            "language": "roman_urdu",
            "emotion": "Neutral",
            "student_id": "2023-ARID-0274",
            "profile": {
                "name": "SHARJEEL AHMED",
                "registration_no": "2023-ARID-0274",
                "cgpa": 3.86,
                "semester": "8",
                "section_code": "A",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "degree_completion")
        reply = res.get("response", "")
        self.assertIn("complete ho jaye gi", reply)
        self.assertIn("fail nahi hai", reply)
        self.assertIn("3.86", reply)
        self.assertIn("12 semesters", reply)
        self.assertNotIn("repeat karna", reply)

    def test_cgpa_assessment_high_roman_urdu(self):
        state: AdvisorState = {
            "query": "kya mera cgpa acha hai",
            "language": "roman_urdu",
            "emotion": "Neutral",
            "student_id": "2023-ARID-0274",
            "profile": {
                "name": "SHARJEEL AHMED",
                "registration_no": "2023-ARID-0274",
                "cgpa": 3.86,
                "semester": "8",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "cgpa")
        reply = res.get("response", "")
        self.assertIn("3.86", reply)
        self.assertIn("shandar", reply.lower())
        self.assertIn("2.50", reply)
        self.assertNotIn("kya meri degree complete", reply.lower())
        self.assertNotIn("student_context", reply.lower())
        # Check concise length (less than 400 characters)
        self.assertLess(len(reply), 400)

    def test_cgpa_assessment_vgpa_typo_english(self):
        state: AdvisorState = {
            "query": "is my vgpa is good",
            "language": "english",
            "emotion": "Neutral",
            "student_id": "2023-ARID-0274",
            "profile": {
                "name": "SHARJEEL AHMED",
                "registration_no": "2023-ARID-0274",
                "cgpa": 3.86,
                "semester": "8",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "cgpa")
        reply = res.get("response", "")
        self.assertIn("3.86", reply)
        self.assertIn("outstanding", reply.lower())
        self.assertIn("2.50", reply)
        self.assertLess(len(reply), 400)

    def test_cgpa_assessment_moderate(self):
        state: AdvisorState = {
            "query": "kya mera cgpa kaisa hai",
            "language": "roman_urdu",
            "emotion": "Neutral",
            "profile": {
                "name": "SHANZA",
                "cgpa": 2.98,
                "semester": "4",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "cgpa")
        reply = res.get("response", "")
        self.assertIn("2.98", reply)
        self.assertIn("safe zone", reply)
        self.assertLess(len(reply), 400)

    def test_semester_cgpa_policy_english(self):
        state: AdvisorState = {
            "query": "what is minimum cgpa requirement for every semester",
            "language": "english",
            "student_id": "2023-ARID-0274",
            "profile": {
                "name": "SHARJEEL AHMED",
                "registration_no": "2023-ARID-0274",
                "cgpa": 3.86,
                "semester": "8",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "cgpa_policy")
        reply = res.get("response", "")
        self.assertIn("8 semesters", reply)
        self.assertIn("12 registered semesters", reply)
        self.assertIn("2.50", reply)
        self.assertIn("1.00", reply)
        self.assertIn("1.25", reply)
        self.assertIn("2.00", reply)
        self.assertNotIn("0.75", reply)

    def test_semester_cgpa_policy_roman_urdu(self):
        state: AdvisorState = {
            "query": "har semester mein kitna cgpa chahiye",
            "language": "roman_urdu",
            "profile": {
                "name": "SHANZA",
                "cgpa": 2.98,
                "semester": "4",
            },
        }
        res = advisor_graph.invoke(state)
        self.assertEqual(res.get("intent"), "cgpa_policy")
        reply = res.get("response", "")
        self.assertIn("8 semesters", reply)
        self.assertIn("12 registered semesters", reply)
        self.assertIn("2.50", reply)
        self.assertIn("1.00", reply)
        self.assertIn("1.25", reply)
        self.assertNotIn("0.75", reply)

    def test_strip_markdown_for_tts_removes_asterisks_and_bullets(self):
        md = (
            "The minimum CGPA is as follows:\n\n"
            "- **Semester 1**: Minimum CGPA is **1.00**\n"
            "- **Semester 8**: Minimum CGPA is **2.50**\n\n"
            "**Degree Duration**: Normal is 8 semesters."
        )
        cleaned = strip_markdown_for_tts(md)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("- ", cleaned)
        self.assertIn("1.00", cleaned)
        self.assertIn("2.50", cleaned)
        self.assertIn("Degree Duration: Normal is 8 semesters.", cleaned)

    def test_strip_markdown_for_tts_handles_roman_urdu(self):
        md = "Aap ka koi course fail nahi hai aur aap ka CGPA **2.98** bohat acha hai!"
        cleaned = strip_markdown_for_tts(md)
        self.assertNotIn("**", cleaned)
        self.assertEqual(cleaned, "Aap ka koi course fail nahi hai aur aap ka CGPA 2.98 bohat acha hai!")


if __name__ == "__main__":
    unittest.main()
