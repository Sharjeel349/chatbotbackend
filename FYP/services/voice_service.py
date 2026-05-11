# import os
# import uuid
# import requests
# import io  # <-- NEW IMPORT
# import speech_recognition as sr
# from gtts import gTTS
# from pydub import AudioSegment  # <-- NEW IMPORT
# from datetime import datetime
#
# # -----------------------------------------
# # Configuration
# # -----------------------------------------
# AUDIO_DIR = "static/audio"
# os.makedirs(AUDIO_DIR, exist_ok=True)
#
# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "phi3"
#
#
# # -----------------------------------------
# # Core Logic
# # -----------------------------------------
# def get_llm_reply(prompt: str) -> str:
#     """Sends the user's text to the local phi3 model."""
#     try:
#         res = requests.post(
#             OLLAMA_URL,
#             json={
#                 "model": OLLAMA_MODEL,
#                 "prompt": prompt,
#                 "stream": False,
#             },
#             timeout=30
#         )
#         res.raise_for_status()
#         return res.json().get("response", "I am having trouble processing that right now.")
#     except Exception as e:
#         print(f"LLM Error: {e}")
#         return "Sorry, my brain (the LLM) is currently offline."
#
#
# def process_voice_chat(session_id: int, audio_file_bytes: bytes, db_conn):
#     """Handles the full Voice -> STT -> LLM -> TTS -> DB pipeline."""
#
#     # 1. Generate unique filenames for this interaction
#     unique_id = uuid.uuid4().hex
#     user_audio_path = os.path.join(AUDIO_DIR, f"user_{unique_id}.wav")
#     bot_audio_path = os.path.join(AUDIO_DIR, f"bot_{unique_id}.mp3")
#
#     # 2. CONVERT and Save the User's Audio File properly
#     # This takes the raw WebM/MP4 bytes from the frontend and converts them to true WAV
#     try:
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_file_bytes))
#         audio_segment.export(user_audio_path, format="wav")
#     except Exception as e:
#         print(f"Audio conversion error: {e}")
#         raise ValueError("Could not convert uploaded audio to WAV.")
#
#     # 3. Speech to Text (STT) - Read the properly formatted file
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(user_audio_path) as source:
#         audio_data = recognizer.record(source)
#         try:
#             user_text = recognizer.recognize_google(audio_data)
#         except sr.UnknownValueError:
#             user_text = "[Unintelligible audio]"
#         except sr.RequestError:
#             user_text = "[STT Service Offline]"
#
#     # 4. Talk to the LLM (if audio was understood)
#     if user_text.startswith("["):
#         bot_reply_text = "I couldn't quite hear you. Could you please repeat that?"
#     else:
#         full_prompt = f"You are a helpful University AI Advisor. A student says: {user_text}"
#         bot_reply_text = get_llm_reply(full_prompt)
#
#     # 5. Text to Speech (TTS) - Generate Bot's voice
#     tts = gTTS(text=bot_reply_text, lang='en', slow=False)
#     tts.save(bot_audio_path)
#
#     # 6. Save EVERYTHING to Database
#     cur = db_conn.cursor()
#     try:
#         query = """
#             INSERT INTO Chat_Message
#             (session_id, sender, message_text, voice_file, timestamp)
#             VALUES (%s, %s, %s, %s, %s)
#         """
#         now = datetime.now()
#
#         cur.execute(query, (session_id, 'User', user_text, user_audio_path, now))
#         cur.execute(query, (session_id, 'Bot', bot_reply_text, bot_audio_path, now))
#
#         db_conn.commit()
#     except Exception as e:
#         db_conn.rollback()
#         print(f"Database Error: {e}")
#     finally:
#         cur.close()
#
#     # 7. Return the data
#     return {
#         "user_text": user_text,
#         "bot_text": bot_reply_text,
#         "user_audio_url": f"/{user_audio_path}",
#         "bot_audio_url": f"/{bot_audio_path}"
#     }


import os
import uuid
import requests
import io
import whisper
import pyttsx3
from pydub import AudioSegment
from datetime import datetime

# -----------------------------------------
# Configuration & Offline Models
# -----------------------------------------
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"

# Load the offline Whisper model into memory globally so it doesn't reload every time
print("Loading offline Whisper model (Base)...")
# fp16=False is recommended for Mac CPUs to avoid warnings
stt_model = whisper.load_model("base")
print("Whisper model loaded and ready!")


# -----------------------------------------
# Core Logic
# -----------------------------------------
def get_llm_reply(prompt: str) -> str:
    """Sends the user's text to the local offline phi3 model."""
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 75,  # Forces it to stay short (approx 2-3 lines)
                    "temperature": 0.3,  # Lower = more professional & less "creative"
                    "stop": ["Student:", "Advisor:", "###", "\n"]  # Cuts off if it tries to roleplay
                }
            },
            timeout=120
        )
        res.raise_for_status()
        return res.json().get("response", "I am having trouble processing that right now.")
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Sorry, my brain (the local LLM) is currently offline."


def process_voice_chat(session_id: int, audio_file_bytes: bytes,emotion: str, db_conn):
    """Handles the full 100% OFFLINE Voice -> STT -> LLM -> TTS -> DB pipeline."""

    # 1. Generate unique filenames
    unique_id = uuid.uuid4().hex
    user_audio_path = os.path.join(AUDIO_DIR, f"user_{unique_id}.wav")

    # We save bot audio as .aiff or .wav because Mac's native TTS creates uncompressed audio
    bot_audio_path = os.path.join(AUDIO_DIR, f"bot_{unique_id}.wav")

    # 2. Convert Web Audio to standard WAV using pydub
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_file_bytes))
        audio_segment.export(user_audio_path, format="wav")
    except Exception as e:
        print(f"Audio conversion error: {e}")
        raise ValueError("Could not convert uploaded audio to WAV.")

    # 3. 100% OFFLINE Speech to Text (Whisper)
    try:
        # Whisper transcribes the file directly from your hard drive
        result = stt_model.transcribe(user_audio_path, fp16=False)
        user_text = result["text"].strip()

        # If it hears nothing, it usually returns an empty string
        if not user_text:
            user_text = "[No speech detected]"
    except Exception as e:
        print(f"Whisper Error: {e}")
        user_text = "[Offline STT Failed]"

    # 4. Talk to the Offline LLM (Ollama)
    if user_text == "[No speech detected]":
        bot_reply_text = "I couldn't quite hear you. Could you please repeat that?"
    else:
        # full_prompt = f"write 2 to 3 line response You are a helpful University AI Advisor if student text is in urdu reply in roman-urdu  A student says: {user_text} and his emotion is {emotion}"
        # full_prompt = f"""Act as a University AI Advisor.Constraint: 2 - 3 lines max.Stay strictlyon - topic.Language
        # Rule: If  user input is Urdu, reply ONLY in Roman Urdu.Otherwise, use English.
        # Input: Student says {user_text}[Emotion: {emotion}]"""
        # full_prompt = f"""You are a professional University AI Advisor.
        #
        # STRICT RULES:
        # 1. Limit response to 2 sentences.
        # 2. If student speaks Urdu (script or Roman), you MUST reply in Roman Urdu (Latin alphabet).
        # 3. Do not mention "If/Then" rules in your reply.
        #
        # ### EXAMPLES
        # Student: "Mera fyp deadline kab hai?"
        # Emotion: Neutral
        # Advisor: Aapki FYP deadline 15 June hai. Koshish karein ke saara kaam waqt par khatam ho jaye.
        #
        # Student: "I need more time for my project."
        # Emotion: Stressed
        # Advisor: I understand you're feeling pressured. Let's discuss a timeline extension to help you manage the workload better.
        #
        # ### REAL TASK
        # Student: "{user_text}"
        # Emotion: {emotion}
        # Advisor:"""
        full_prompt = f"""You are a University AI Advisor.

        Rules:
        - Max 2 sentences
        - Urdu input → Roman Urdu reply
        - Stay relevant and helpful
        
        Examples:
        Student: I want to freeze my semester.
        Advisor: i understand but try to manage but id you still want  semester freeze contact datacell but make sure you have not freeze more than 2
        Student: میرا گریڈ بہت گندا آیا ہے، میں کیا کروں
        Advisor: mein smjh sakta hon ap or zada mehnat kro demotivate nhi hona 
        
        Real Task: Student: "{user_text}" Emotion: {emotion}
        Advisor:"""
        bot_reply_text = get_llm_reply(full_prompt)

    # 5. 100% OFFLINE Text to Speech (Mac Native Voice via pyttsx3)
    try:
        engine = pyttsx3.init()
        # This triggers your Mac's internal text-to-speech engine and saves it to a file
        engine.save_to_file(bot_reply_text, bot_audio_path)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

    # 6. Save to Database
    cur = db_conn.cursor()
    try:
        query = """
            INSERT INTO Chat_Message 
            (session_id, sender, message_text, voice_file, emotion)
            VALUES (%s, %s, %s, %s, %s)
        """
        now = datetime.now()

        cur.execute(query, (session_id, 'User', user_text, user_audio_path,emotion))
        cur.execute(query, (session_id, 'Bot', bot_reply_text, bot_audio_path,"Neutral"))

        db_conn.commit()
    except Exception as e:
        db_conn.rollback()
        print(f"Database Error: {e}")
    finally:
        cur.close()

    # 7. Return the data
    return {
        "user_text": user_text,
        "bot_text": bot_reply_text,
        "user_audio_url": f"/{user_audio_path}",
        "bot_audio_url": f"/{bot_audio_path}",
        "user_emotion": emotion,
    }