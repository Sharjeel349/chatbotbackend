from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = BASE_DIR / "static" / "audio" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_TEXT = (
    "Hi, I'm your university advisor. How can I help you? "
    "Assalamu alaikum, mein ap ka university advisor hon, mein ap ki kaisy madad kr sakta ho"
)

# Curated metadata for popular and recommended voices
CURATED_VOICE_META: Dict[str, Dict[str, Any]] = {
    "Lekha": {
        "display_name": "Lekha (Roman Urdu & Hindi) ★ Recommended",
        "gender": "Female",
        "category": "Urdu & Roman Urdu",
        "is_recommended": True,
        "default": True,
        "locale": "hi_IN",
        "language": "Hindi / Roman Urdu",
    },
    "Rishi": {
        "display_name": "Rishi (Indian English / South Asian)",
        "gender": "Male",
        "category": "South Asian English",
        "is_recommended": True,
        "locale": "en_IN",
        "language": "English (India)",
    },
    "Samantha": {
        "display_name": "Samantha (US English - Clear Female)",
        "gender": "Female",
        "category": "English (US)",
        "is_recommended": True,
        "locale": "en_US",
        "language": "English (United States)",
    },
    "Daniel": {
        "display_name": "Daniel (UK English - Clear Male)",
        "gender": "Male",
        "category": "English (UK)",
        "is_recommended": True,
        "locale": "en_GB",
        "language": "English (United Kingdom)",
    },
    "Karen": {
        "display_name": "Karen (Australian English - Female)",
        "gender": "Female",
        "category": "English (Australia)",
        "is_recommended": True,
        "locale": "en_AU",
        "language": "English (Australia)",
    },
    "Moira": {
        "display_name": "Moira (Irish English - Female)",
        "gender": "Female",
        "category": "English (Ireland)",
        "is_recommended": True,
        "locale": "en_IE",
        "language": "English (Ireland)",
    },
    "Tessa": {
        "display_name": "Tessa (South African English - Female)",
        "gender": "Female",
        "category": "English (South Africa)",
        "is_recommended": True,
        "locale": "en_ZA",
        "language": "English (South Africa)",
    },
    "Fred": {
        "display_name": "Fred (Classic US Male)",
        "gender": "Male",
        "category": "English (US)",
        "is_recommended": False,
        "locale": "en_US",
        "language": "English (United States)",
    },
    "Albert": {
        "display_name": "Albert (US English)",
        "gender": "Male",
        "category": "English (US)",
        "is_recommended": False,
        "locale": "en_US",
        "language": "English (United States)",
    },
}

_tts_lock = threading.Lock()


def _is_say_available() -> bool:
    """Check if macOS native 'say' utility is available."""
    return shutil.which("say") is not None


@lru_cache(maxsize=1)
def get_system_voices() -> List[Dict[str, Any]]:
    """Enumerate system TTS voices with structured metadata."""
    voices: List[Dict[str, Any]] = []

    # 1. On macOS, parse native say voices for highest fidelity
    if _is_say_available():
        try:
            cmd_output = subprocess.check_output(["say", "-v", "?"], text=True, timeout=5)
            for line in cmd_output.strip().splitlines():
                match = re.match(r"^([^\t]+?)\s{2,}([a-z]{2}_[A-Z]{2})\s+#\s*(.*)$", line)
                if not match:
                    continue
                name, lang, example = match.groups()
                name = name.strip()
                lang = lang.strip()

                meta = CURATED_VOICE_META.get(name, {})
                display_name = meta.get("display_name", f"{name} ({lang})")
                gender = meta.get("gender") or ("Female" if any(w in name.lower() for w in ("alice", "alva", "anna", "amelie", "carmit", "karen", "lekha", "moira", "samantha", "tessa", "kathy")) else "Male")
                is_recommended = meta.get("is_recommended", False)
                category = meta.get("category", "English" if lang.startswith("en") else ("Urdu & South Asian" if lang.startswith("hi") or lang.startswith("ur") else "Other Languages"))

                voices.append({
                    "id": name,
                    "name": name,
                    "display_name": display_name,
                    "language": meta.get("language", lang),
                    "locale": meta.get("locale", lang),
                    "gender": gender,
                    "is_recommended": is_recommended,
                    "category": category,
                    "sample_url": f"/api/voices/preview?voice={name}",
                })
        except Exception as exc:
            logger.warning("Could not list macOS voices via say: %s", exc)

    # 2. Fallback to pyttsx3 if no voices found or non-macOS
    if not voices:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            py_voices = engine.getProperty("voices")
            for v in py_voices:
                vid = str(v.id)
                vname = str(v.name)
                langs = getattr(v, "languages", [])
                lang_code = langs[0] if langs else "en_US"

                meta = CURATED_VOICE_META.get(vname, {})
                display_name = meta.get("display_name", f"{vname} ({lang_code})")
                gender = meta.get("gender") or str(getattr(v, "gender", "Unknown")).replace("VoiceGender", "")
                is_recommended = meta.get("is_recommended", False)

                voices.append({
                    "id": vid,
                    "name": vname,
                    "display_name": display_name,
                    "language": meta.get("language", str(lang_code)),
                    "locale": meta.get("locale", str(lang_code)),
                    "gender": gender,
                    "is_recommended": is_recommended,
                    "category": meta.get("category", "General"),
                    "sample_url": f"/api/voices/preview?voice={vid}",
                })
        except Exception as exc:
            logger.error("Could not enumerate pyttsx3 voices: %s", exc)

    # Sort recommended voices first, then alphabetically by category and name
    voices.sort(key=lambda x: (not x["is_recommended"], x["category"], x["name"]))
    return voices


def get_available_voices_response() -> Dict[str, Any]:
    """Return complete structured voice payload for frontend dropdowns."""
    all_voices = get_system_voices()
    recommended = [v for v in all_voices if v.get("is_recommended")]

    # Find the default voice (Lekha for bilingual Urdu/English advisor, or first recommended)
    default_voice = "Lekha" if any(v["name"] == "Lekha" for v in all_voices) else (recommended[0]["id"] if recommended else (all_voices[0]["id"] if all_voices else ""))

    return {
        "status": "success",
        "total": len(all_voices),
        "default_voice": default_voice,
        "default_sample_text": DEFAULT_SAMPLE_TEXT,
        "recommended_voices": recommended,
        "all_voices": all_voices,
    }


def synthesize_sample(voice: str, text: Optional[str] = None) -> Tuple[Path, str, int]:
    """
    Synthesize audio for a given voice and sample text.
    Returns (file_path, audio_url, duration_ms).
    Uses caching so repeated requests return instantaneously.
    """
    sample_text = (text or "").strip() or DEFAULT_SAMPLE_TEXT
    safe_voice = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice)
    text_hash = hashlib.md5(sample_text.encode("utf-8")).hexdigest()[:10]
    out_filename = f"sample_{safe_voice}_{text_hash}.wav"
    out_path = SAMPLES_DIR / out_filename
    audio_url = f"/api/static/audio/samples/{out_filename}"

    # Cache hit check
    if out_path.exists() and out_path.stat().st_size > 4096:
        logger.debug("Cache hit for voice preview: %s", out_path)
        return out_path, audio_url, 0

    started = time.perf_counter()
    with _tts_lock:
        # Double check after acquiring lock in case another request synthesized it
        if out_path.exists() and out_path.stat().st_size > 4096:
            return out_path, audio_url, 0

        temp_raw = SAMPLES_DIR / f"temp_{safe_voice}_{text_hash}.aiff"

        # Attempt 1: Native macOS 'say' command (highest audio quality & perfect pronunciation)
        if _is_say_available():
            try:
                cmd = ["say", "-v", voice, "-o", str(temp_raw), sample_text]
                subprocess.run(cmd, check=True, timeout=15)

                if temp_raw.exists() and temp_raw.stat().st_size > 1024:
                    segment = AudioSegment.from_file(str(temp_raw))
                    segment.export(str(out_path), format="wav")
                    temp_raw.unlink(missing_ok=True)
                    elapsed = round((time.perf_counter() - started) * 1000)
                    return out_path, audio_url, elapsed
            except Exception as exc:
                logger.warning("say command failed for voice '%s': %s; trying pyttsx3", voice, exc)
                temp_raw.unlink(missing_ok=True)

        # Attempt 2: pyttsx3 fallback
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            try:
                engine.setProperty("voice", voice)
            except Exception:
                logger.warning("Could not set voice property to '%s'", voice)

            engine.save_to_file(sample_text, str(temp_raw))
            engine.runAndWait()

            if temp_raw.exists() and temp_raw.stat().st_size > 4096:
                segment = AudioSegment.from_file(str(temp_raw))
                segment.export(str(out_path), format="wav")
                temp_raw.unlink(missing_ok=True)
                elapsed = round((time.perf_counter() - started) * 1000)
                return out_path, audio_url, elapsed
            else:
                # If voice failed, try fallback voice Lekha or Samantha
                engine.setProperty("voice", "Lekha")
                engine.save_to_file(sample_text, str(temp_raw))
                engine.runAndWait()
                if temp_raw.exists() and temp_raw.stat().st_size > 1024:
                    segment = AudioSegment.from_file(str(temp_raw))
                    segment.export(str(out_path), format="wav")
                    temp_raw.unlink(missing_ok=True)
                    elapsed = round((time.perf_counter() - started) * 1000)
                    return out_path, audio_url, elapsed
        except Exception as exc:
            logger.error("TTS synthesis failed completely for '%s': %s", voice, exc)
            temp_raw.unlink(missing_ok=True)
            raise RuntimeError(f"Could not generate voice preview for '{voice}': {exc}")

    if not out_path.exists():
        raise RuntimeError(f"Failed to generate voice preview for voice '{voice}'")

    elapsed = round((time.perf_counter() - started) * 1000)
    return out_path, audio_url, elapsed


def prewarm_default_voices() -> None:
    """Pre-generate preview audio for recommended voices during application startup."""
    try:
        recommended = ["Lekha", "Samantha", "Daniel", "Rishi"]
        for voice_name in recommended:
            try:
                synthesize_sample(voice_name, DEFAULT_SAMPLE_TEXT)
                logger.info("Pre-warmed voice sample for: %s", voice_name)
            except Exception as e:
                logger.debug("Could not prewarm voice %s: %s", voice_name, e)
    except Exception as exc:
        logger.warning("Pre-warming voices error: %s", exc)
