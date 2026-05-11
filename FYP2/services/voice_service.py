#
# import os
# import uuid
# import requests
# import io
# import whisper
# import pyttsx3
# from pydub import AudioSegment
# from datetime import datetime
#
# # -----------------------------------------
# # Configuration & Offline Models
# # -----------------------------------------
# AUDIO_DIR = "static/audio"
# os.makedirs(AUDIO_DIR, exist_ok=True)
#
# OLLAMA_URL = "http://localhost:11434/api/generate"
# # OLLAMA_MODEL = "gemma2:2b"
# OLLAMA_MODEL = "mistral"
# # Load the offline Whisper model into memory globally so it doesn't reload every time
# print("Loading offline Whisper model (Base)...")
# # fp16=False is recommended for Mac CPUs to avoid warnings
# stt_model = whisper.load_model("base")
# print("Whisper model loaded and ready!")
#
#
# def get_latest_emotion(db_conn) -> str:
#     """Fetches the absolute latest EEG mind state from the database."""
#     cur = db_conn.cursor()
#     try:
#         # Order by ID descending to get the newest row
#         query = "SELECT mind_State FROM EEG_State ORDER BY eeg_state_id DESC LIMIT 1"
#         cur.execute(query)
#         result = cur.fetchone()
#
#         if result:
#             return result[0]  # Returns 'Neutral', 'CALM', or 'STRESSED'
#         else:
#             return "Neutral"  # Default fallback if the table is completely empty
#
#     except Exception as e:
#         print(f"Error fetching latest emotion: {e}")
#         return "Neutral"  # Safe fallback if the database query fails
#     finally:
#         cur.close()
#
#
# # -----------------------------------------
# # Core Logic
# # -----------------------------------------
# def get_llm_reply(prompt: str) -> str:
#     """Sends the user's text to the local offline phi3 model."""
#     try:
#         res = requests.post(
#             OLLAMA_URL,
#             json={
#                 "model": OLLAMA_MODEL,
#                 "prompt": prompt,
#                 "stream": False,
#             },
#             timeout=120
#         )
#         res.raise_for_status()
#         return res.json().get("response", "I am having trouble processing that right now.")
#     except Exception as e:
#         print(f"LLM Error: {e}")
#         return "Sorry, my brain (the local LLM) is currently offline."
#
#
# def process_voice_chat(session_id: int, audio_file_bytes: bytes,emotion: str, db_conn):
#     """Handles the full 100% OFFLINE Voice -> STT -> LLM -> TTS -> DB pipeline."""
#
#     # 1. Generate unique filenames
#     unique_id = uuid.uuid4().hex
#     user_audio_path = os.path.join(AUDIO_DIR, f"user_{unique_id}.wav")
#
#     # We save bot audio as .aiff or .wav because Mac's native TTS creates uncompressed audio
#     bot_audio_path = os.path.join(AUDIO_DIR, f"bot_{unique_id}.wav")
#
#     # 2. Convert Web Audio to standard WAV using pydub
#     try:
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_file_bytes))
#         audio_segment.export(user_audio_path, format="wav")
#     except Exception as e:
#         print(f"Audio conversion error: {e}")
#         raise ValueError("Could not convert uploaded audio to WAV.")
#
#     # 3. 100% OFFLINE Speech to Text (Whisper)
#     try:
#         # Whisper transcribes the file directly from your hard drive
#         result = stt_model.transcribe(user_audio_path, fp16=False)
#         user_text = result["text"].strip()
#
#         # If it hears nothing, it usually returns an empty string
#         if not user_text:
#             user_text = "[No speech detected]"
#     except Exception as e:
#         print(f"Whisper Error: {e}")
#         user_text = "[Offline STT Failed]"
#
#     emotion=get_latest_emotion(db_conn)
#
#     # 4. Talk to the Offline LLM (Ollama)
#     if user_text == "[No speech detected]":
#         bot_reply_text = "I couldn't quite hear you. Could you please repeat that?"
#     else:
#
#         full_prompt = f"""
#         You are a University AI Advisor.
#
#         STRICT RULES (NO EXCEPTIONS):
#         - Output must be in Roman Urdu only (English letters)
#         - NEVER use Urdu script
#         - Max 2 short sentences
#         - Keep response simple and helpful
#
#         Student: "{user_text}"
#         Emotion: {emotion}
#         Advisor:
#         """
#         bot_reply_text = get_llm_reply(full_prompt)
#
#     # 5. 100% OFFLINE Text to Speech (Mac Native Voice via pyttsx3)
#     try:
#         engine = pyttsx3.init()
#         # This triggers your Mac's internal text-to-speech engine and saves it to a file
#         engine.save_to_file(bot_reply_text, bot_audio_path)
#         engine.runAndWait()
#     except Exception as e:
#         print(f"TTS Error: {e}")
#
#     # 6. Save to Database
#     cur = db_conn.cursor()
#     try:
#         emotions = get_latest_emotion(db_conn)
#         query = """
#             INSERT INTO Chat_Message
#             (session_id, sender, message_text, voice_file, emotion)
#             VALUES (%s, %s, %s, %s, %s)
#         """
#         now = datetime.now()
#         print(emotions)
#         print(emotion)
#         cur.execute(query, (session_id, 'User', user_text, user_audio_path,emotions))
#         cur.execute(query, (session_id, 'Bot', bot_reply_text, bot_audio_path,"Neutral"))
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
#         "bot_audio_url": f"/{bot_audio_path}",
#         "user_emotion": emotion,
#     }


import os
import uuid
import requests
import io
import whisper
import pyttsx3
from pydub import AudioSegment


# -----------------------------------------
# Configuration & Offline Models
# -----------------------------------------
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_MODEL = "mistral"
OLLAMA_MODEL = "gemma2:2b"

print("Loading offline Whisper model (Base)...")
stt_model = whisper.load_model("base")
print("Whisper model loaded and ready!")


# -----------------------------------------
# Emotion Fetch (FIXED)
# -----------------------------------------
def get_latest_emotion(db_conn) -> str:
    """Fetch latest EEG emotion safely."""
    cur = db_conn.cursor()
    try:
        query = """
            SELECT mind_State 
            FROM EEG_State 
            ORDER BY eeg_state_id DESC 
            LIMIT 1
        """
        cur.execute(query)
        result = cur.fetchone()

        print("EEG Query Result:", result)  # DEBUG

        if result:
            emotion = result.get('mind_state') or result.get('mind_State')
            if emotion:
                return emotion

        return "Neutral"

    except Exception as e:
        print(f"Error fetching emotion: {e}")
        return "Neutral"
    finally:
        cur.close()


# -----------------------------------------
# LLM Call
# -----------------------------------------
def get_llm_reply(prompt: str) -> str:
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120
        )
        res.raise_for_status()
        return res.json().get("response", "System issue. Try again.")
    except Exception as e:
        print(f"LLM Error: {e}")
        return "LLM offline hai."


# -----------------------------------------
# MAIN PIPELINE (FIXED)
# -----------------------------------------
def process_voice_chat(session_id: int, audio_file_bytes: bytes, db_conn):
    unique_id = uuid.uuid4().hex
    user_audio_path = os.path.join(AUDIO_DIR, f"user_{unique_id}.wav")
    bot_audio_path = os.path.join(AUDIO_DIR, f"bot_{unique_id}.wav")

    # -----------------------------------------
    # 1. Convert Audio
    # -----------------------------------------
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_file_bytes))
        audio_segment.export(user_audio_path, format="wav")
    except Exception as e:
        print(f"Audio conversion error: {e}")
        raise ValueError("Audio conversion failed")

    # -----------------------------------------
    # 2. Speech to Text (Whisper)
    # -----------------------------------------
    try:
        result = stt_model.transcribe(user_audio_path, fp16=False)
        user_text = result["text"].strip()

        if not user_text:
            user_text = "[No speech detected]"

    except Exception as e:
        print(f"Whisper Error: {e}")
        user_text = "[STT Failed]"

    # -----------------------------------------
    # 3. Get Emotion ONCE (FIXED)
    # -----------------------------------------
    latest_emotion = get_latest_emotion(db_conn)
    print("Latest Emotion Used:", latest_emotion)

    # -----------------------------------------
    # 4. LLM Response
    # -----------------------------------------
    if user_text == "[No speech detected]":
        bot_reply_text = "Mujhe awaaz clear nahi aayi, dobara bolain."
    else:
        full_prompt = f"""
You are a University AI Advisor.

Rules:
- Reply in English only
- Max 2 sentences
- Helpful and simple

Student: "{user_text}"
Emotion: {latest_emotion}
Advisor:
"""
        bot_reply_text = get_llm_reply(full_prompt)
        romanprompt=f""" convert it in to roman urdu{bot_reply_text}"""
        bot_reply_text = get_llm_reply(romanprompt)

    # -----------------------------------------
    # 5. Text to Speech
    # -----------------------------------------
    try:
        engine = pyttsx3.init()
        engine.save_to_file(bot_reply_text, bot_audio_path)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

    # -----------------------------------------
    # 6. Save to DB (FIXED)
    # -----------------------------------------
    cur = db_conn.cursor()
    try:
        query = """
            INSERT INTO Chat_Message
            (session_id, sender, message_text, voice_file, emotion)
            VALUES (%s, %s, %s, %s, %s)
        """

        cur.execute(query, (
            session_id, 'User', user_text, user_audio_path, latest_emotion
        ))

        cur.execute(query, (
            session_id, 'Bot', bot_reply_text, bot_audio_path, "Neutral"
        ))

        db_conn.commit()

    except Exception as e:
        db_conn.rollback()
        print(f"Database Error: {e}")
    finally:
        cur.close()

    # -----------------------------------------
    # 7. Response
    # -----------------------------------------
    return {
        "user_text": user_text,
        "bot_text": bot_reply_text,
        "user_audio_url": f"/{user_audio_path}",
        "bot_audio_url": f"/{bot_audio_path}",
        "user_emotion": latest_emotion,
    }