from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyttsx3
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


def _resolve_pyttsx3_voice(engine: pyttsx3.Engine, voice_name_or_id: str) -> Optional[str]:
    """Find the best matching pyttsx3 voice ID for the given voice name or ID."""
    target = (voice_name_or_id or "").strip().lower()
    if not target:
        return None

    voices = engine.getProperty("voices") or []
    # 1. Exact ID match
    for v in voices:
        if str(v.id).lower() == target:
            return v.id

    # 2. Exact Name match
    for v in voices:
        if str(v.name).lower() == target:
            return v.id

    # 3. Substring match in ID or Name
    for v in voices:
        vid_lower = str(v.id).lower()
        vname_lower = str(v.name).lower()
        if target in vid_lower or target in vname_lower:
            return v.id

    # 4. If target is Lekha/Urdu/Hindi, find any Hindi/Urdu voice
    if any(k in target for k in ("lekha", "urdu", "hindi", "roman_urdu")):
        for v in voices:
            vid_lower = str(v.id).lower()
            langs = [str(l).lower() for l in getattr(v, "languages", [])]
            if "hi-in" in vid_lower or "lekha" in vid_lower or any("hi" in l or "ur" in l for l in langs):
                return v.id

    # 5. Fallback to English voice if Samantha requested
    if "samantha" in target:
        for v in voices:
            vid_lower = str(v.id).lower()
            vname_lower = str(v.name).lower()
            if "samantha" in vid_lower or "samantha" in vname_lower:
                return v.id

    return None


@lru_cache(maxsize=1)
def get_system_voices() -> List[Dict[str, Any]]:
    """Enumerate system TTS voices completely via pyttsx3 with structured metadata."""
    voices: List[Dict[str, Any]] = []
    try:
        engine = pyttsx3.init()
        py_voices = engine.getProperty("voices") or []
        for v in py_voices:
            vid = str(getattr(v, "id", "")).strip()
            vname = str(getattr(v, "name", "")).strip()
            if not vid and not vname:
                continue

            langs = [str(l).lower() for l in getattr(v, "languages", [])]
            lang_code = langs[0] if langs else "en_US"

            # Match curated metadata by voice name or ID
            meta = {}
            for key, val in CURATED_VOICE_META.items():
                if key.lower() == vname.lower() or key.lower() in vid.lower():
                    meta = val
                    break

            display_name = meta.get("display_name") or f"{vname} ({lang_code})"
            gender = meta.get("gender") or str(getattr(v, "gender", "Unknown")).replace("VoiceGender", "")
            if gender in ("Unknown", "None", ""):
                gender = (
                    "Female"
                    if any(
                        w in vname.lower()
                        for w in (
                            "alice", "alva", "anna", "amelie", "carmit",
                            "karen", "lekha", "moira", "samantha", "tessa", "kathy"
                        )
                    )
                    else "Male"
                )

            is_recommended = meta.get("is_recommended", False)
            category = meta.get("category")
            if not category:
                if any(x in lang_code for x in ("hi", "ur", "pa")) or any(x in vid.lower() for x in ("hi-in", "lekha")):
                    category = "Urdu & South Asian"
                elif any(x in lang_code for x in ("en", "us", "gb", "au", "in")):
                    category = "English"
                else:
                    category = "Other Languages"

            voices.append({
                "id": vid,
                "name": vname,
                "display_name": display_name,
                "language": meta.get("language", str(lang_code)),
                "locale": meta.get("locale", str(lang_code)),
                "gender": gender,
                "is_recommended": is_recommended,
                "category": category,
                "sample_url": f"/api/voices/preview?voice={vname}",
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


def strip_markdown_for_tts(text: str) -> str:
    """
    Strips markdown formatting, symbols, and list bullets from text so
    that Text-To-Speech engines speak smoothly without pronouncing asterisks, hashes, etc.
    """
    if not text:
        return ""
    # Links [label](url) -> label
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove markdown header markers at line start (#, ##, etc.)
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.MULTILINE)
    # Remove bold/italic/strikethrough/code markers (*, _, ~, `)
    s = re.sub(r"[*_~`]+", "", s)
    # Remove bullet markers (- , * , + , • ) at line start
    s = re.sub(r"^\s*[-*+•]\s+", "", s, flags=re.MULTILINE)
    # For line breaks: if line ends without terminal punctuation, append a period
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    cleaned_lines = []
    for line in lines:
        if line and line[-1] not in (".", "!", "?", ":", ","):
            cleaned_lines.append(line + ".")
        else:
            cleaned_lines.append(line)
    s = " ".join(cleaned_lines)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _synthesize_with_pyttsx3(voice_id: Optional[str], text: str, temp_raw: Path) -> bool:
    """
    Synthesize audio using pyttsx3.
    On macOS, pyttsx3 uses AppKit's NSSpeechSynthesizer. Pumping the NSRunLoop until
    synthesis finishes ensures 100% complete audio output for utterances of any length.
    On other platforms (Windows/Linux), standard pyttsx3 save_to_file and runAndWait are used.
    """
    # 1. On macOS (in-process pyttsx3 NSSS engine driver)
    try:
        from AppKit import NSSpeechSynthesizer
        from Foundation import NSURL, NSRunLoop, NSDefaultRunLoopMode, NSDate

        synth = (
            NSSpeechSynthesizer.alloc().initWithVoice_(voice_id)
            if voice_id
            else NSSpeechSynthesizer.alloc().init()
        )
        synth.setRate_(160)
        url = NSURL.fileURLWithPath_(str(temp_raw))
        temp_raw.unlink(missing_ok=True)
        synth.startSpeakingString_toURL_(text, url)

        runloop = NSRunLoop.currentRunLoop()
        deadline = time.time() + 30.0
        while synth.isSpeaking() and time.time() < deadline:
            runloop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))

        if temp_raw.exists() and temp_raw.stat().st_size > 1024:
            return True
    except Exception as exc:
        logger.debug("Direct AppKit pyttsx3 synthesis exception: %s", exc)

    # 2. Standard pyttsx3 engine fallback
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        if voice_id:
            try:
                engine.setProperty("voice", voice_id)
            except Exception:
                pass
        temp_raw.unlink(missing_ok=True)
        engine.save_to_file(text, str(temp_raw))
        engine.runAndWait()
        return temp_raw.exists() and temp_raw.stat().st_size > 1024
    except Exception as exc:
        logger.error("Standard pyttsx3 save_to_file failed: %s", exc)
        return False


def synthesize_sample(voice: str, text: Optional[str] = None) -> Tuple[Path, str, int]:
    """
    Synthesize audio using pyttsx3 for a given voice and sample text.
    Returns (file_path, audio_url, duration_ms).
    Uses caching so repeated requests return instantaneously.
    """
    raw_text = (text or "").strip() or DEFAULT_SAMPLE_TEXT
    sample_text = strip_markdown_for_tts(raw_text)
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

        # Resolve pyttsx3 voice ID
        engine = pyttsx3.init()
        voice_id = _resolve_pyttsx3_voice(engine, voice)

        # Synthesize via pyttsx3
        ok = _synthesize_with_pyttsx3(voice_id, sample_text, temp_raw)
        if not ok:
            # Fallback to Lekha or Samantha
            fallback_id = _resolve_pyttsx3_voice(engine, "Lekha") or _resolve_pyttsx3_voice(engine, "Samantha")
            ok = _synthesize_with_pyttsx3(fallback_id, sample_text, temp_raw)

        if ok and temp_raw.exists() and temp_raw.stat().st_size > 1024:
            segment = AudioSegment.from_file(str(temp_raw))
            segment.export(str(out_path), format="wav")
            temp_raw.unlink(missing_ok=True)
            elapsed = round((time.perf_counter() - started) * 1000)
            return out_path, audio_url, elapsed

    if not out_path.exists() or out_path.stat().st_size <= 1024:
        raise RuntimeError(f"Failed to generate audio with pyttsx3 for voice '{voice}'")

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
