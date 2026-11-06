from __future__ import annotations

import datetime
from pathlib import Path
import sys
import threading
import time
from typing import Any

# Ensure project root is on sys.path so FYP2 can be imported directly when running this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from FYP2.core.config import DB_CONFIG

# Default Muse channel names
DEFAULT_MUSE_5NODE_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]

# Thread-safe in-memory stream buffers for live sessions
_stream_lock = threading.Lock()
_session_live_buffers: dict[Any, list[list[float]]] = {}
_session_30s_buffers: dict[Any, list[list[float]]] = {}
_session_last_30s_save: dict[Any, float] = {}


def get_channel_names(inlet):
    ch_names = []
    try:
        info = inlet.info()
        ch = info.desc().child("channels").child("channel")
        while not ch.empty():
            label = ch.child_value("label")
            if label:
                ch_names.append(label)
            ch = ch.next_sibling("channel")
    except Exception:
        pass

    if not ch_names:
        count = inlet.info().channel_count() if hasattr(inlet, "info") else 5
        if count <= 5:
            ch_names = DEFAULT_MUSE_5NODE_CHANNELS[:count]
        else:
            ch_names = [f"CH{i+1}" for i in range(count)]

    return ch_names


def save_live_rows_to_db(rows: list[dict], session_id: int | None = None) -> None:
    """Inserts continuous, dense rows into eeg_live_data so graphs render smoothly."""
    if not rows:
        return
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO eeg_live_data 
            (session_id, alpha_power, beta_power, theta_power, ratio, mind_state, tp9, af7, af8, tp10, timestamp)
            VALUES %s;
        """
        records = [
            (
                session_id,
                r.get("alpha", 0.0),
                r.get("beta", 0.0),
                r.get("theta", 0.0),
                r.get("ratio", 0.0),
                r.get("mind_state", "Neutral"),
                r.get("tp9"),
                r.get("af7"),
                r.get("af8"),
                r.get("tp10"),
                r.get("timestamp", datetime.datetime.now()),
            )
            for r in rows
        ]
        execute_values(cursor, insert_query, records)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"📊 [eeg_live_data] Inserted {len(rows)} continuous rows (session_id={session_id})")
    except Exception as e:
        print(f"❌ DATABASE ERROR (eeg_live_data): {e}")


def save_30s_emotion_to_db(mind_state: str, session_id: int | None = None) -> None:
    """Inserts a single aggregated emotion row into eeg_state every 30 seconds."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO eeg_state (mind_state, session_id) VALUES (%s, %s);"
        cursor.execute(query, (mind_state, session_id))
        conn.commit()
        cursor.close()
        conn.close()
        print("\n" + "=" * 55)
        print(f"🧠 [30s EEG_State SAVED]: {mind_state} (session_id={session_id})")
        print("=" * 55 + "\n")
    except Exception as e:
        print(f"❌ DATABASE ERROR (eeg_state): {e}")


def save_to_db(mind_state, session_id=None, metrics=None):
    """Backwards-compatible helper for legacy callers."""
    if metrics:
        save_live_rows_to_db([metrics], session_id=session_id)
    save_30s_emotion_to_db(mind_state, session_id=session_id)


def analyze_eeg_buffer(buffer, sample_rate=256):
    """Analyze single EEG buffer chunk using FFT power spectral density."""
    if not buffer:
        return {"mind_state": "Neutral", "alpha": 0.0, "beta": 0.0, "theta": 0.0, "ratio": 0.0}
    data = np.asarray(buffer, dtype=float)
    if data.ndim != 2 or data.shape[0] < 16:
        return {"mind_state": "Neutral", "alpha": 0.0, "beta": 0.0, "theta": 0.0, "ratio": 0.0}
    num_channels = min(5, data.shape[1])
    eeg = data[:, :num_channels]
    eeg -= np.mean(eeg, axis=0, keepdims=True)
    eeg *= np.hanning(eeg.shape[0])[:, None]

    frequencies = np.fft.rfftfreq(eeg.shape[0], d=1 / sample_rate)
    power = np.mean(np.abs(np.fft.rfft(eeg, axis=0)) ** 2, axis=1)

    def band_power(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    ratio = float(beta / max(alpha, 1e-9))

    if ratio > 1.6:
        state = "STRESSED"
    elif ratio < 0.8:
        state = "CALM"
    else:
        state = "Neutral"

    return {
        "mind_state": state,
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "theta": round(theta, 4),
        "ratio": round(ratio, 4),
    }


def compute_sliding_live_metrics(
    buffer: list[list[float]],
    window_size: int = 128,
    step_size: int = 32,
    sample_rate: int = 256,
) -> tuple[list[dict], list[list[float]]]:
    """
    Computes smooth, overlapping spectral metrics (Theta, Alpha, Beta, Ratio)
    for every step across the buffer so eeg_live_data will produce a continuous smooth graph.
    Returns:
        (live_rows, remaining_tail)
    """
    if not buffer or len(buffer) < window_size:
        return [], buffer

    data = np.asarray(buffer, dtype=float)
    num_channels = min(5, data.shape[1])
    eeg_all = data[:, :num_channels]

    live_rows = []
    now = datetime.datetime.now()
    total_samples = len(eeg_all)
    idx = 0

    while idx + window_size <= total_samples:
        window = eeg_all[idx : idx + window_size]

        w_centered = window - np.mean(window, axis=0, keepdims=True)
        w_windowed = w_centered * np.hanning(window_size)[:, None]

        freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
        power = np.mean(np.abs(np.fft.rfft(w_windowed, axis=0)) ** 2, axis=1)

        def band_power(low, high):
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                return 0.0
            return float(np.sum(power[mask]))

        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta = band_power(13, 30)
        ratio = float(beta / max(alpha, 1e-9))

        if ratio > 1.6:
            state = "STRESSED"
        elif ratio < 0.8:
            state = "CALM"
        else:
            state = "Neutral"

        mean_ch = np.mean(window, axis=0)
        tp9 = float(mean_ch[0]) if num_channels > 0 else None
        af7 = float(mean_ch[1]) if num_channels > 1 else None
        af8 = float(mean_ch[2]) if num_channels > 2 else None
        tp10 = float(mean_ch[3]) if num_channels > 3 else None

        offset_sec = (idx - total_samples) / float(sample_rate)
        row_time = now + datetime.timedelta(seconds=offset_sec)

        live_rows.append({
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "theta": round(theta, 4),
            "ratio": round(ratio, 4),
            "mind_state": state,
            "tp9": round(tp9, 2) if tp9 is not None else None,
            "af7": round(af7, 2) if af7 is not None else None,
            "af8": round(af8, 2) if af8 is not None else None,
            "tp10": round(tp10, 2) if tp10 is not None else None,
            "timestamp": row_time,
        })

        idx += step_size

    remaining_tail = eeg_all[idx:].tolist()
    return live_rows, remaining_tail


def analyze_30s_data(buffer, sample_rate=256):
    """Calculates overall emotion state across a 30-second EEG window."""
    if not buffer:
        return "Neutral"
    data = np.asarray(buffer, dtype=float)
    if data.ndim != 2 or data.shape[0] < sample_rate * 3:
        return "Neutral"
    num_channels = min(5, data.shape[1])
    eeg = data[:, :num_channels]
    eeg -= np.mean(eeg, axis=0, keepdims=True)
    eeg *= np.hanning(eeg.shape[0])[:, None]

    frequencies = np.fft.rfftfreq(eeg.shape[0], d=1 / sample_rate)
    power = np.mean(np.abs(np.fft.rfft(eeg, axis=0)) ** 2, axis=1)

    def band_power(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    ratio = beta / max(alpha, 1e-9)

    if ratio > 1.6:
        return "STRESSED"
    elif ratio < 0.8:
        return "CALM"
    return "Neutral"


def process_and_save_eeg_stream(buffer: list, session_id: int | None = None) -> dict:
    """
    Entry point for the FastAPI EEG stream route (/api/eeg/stream):
    1. LIVE DATA: Inserts every single row of computed live wave metrics into eeg_live_data
       using an overlapping sliding window so the graph will be dense and smooth.
    2. EEG STATE: Inserts the aggregated emotion into eeg_state ONLY every 30 seconds.
    """
    if not buffer:
        return {"status": "empty"}

    with _stream_lock:
        if session_id not in _session_live_buffers:
            _session_live_buffers[session_id] = []
            _session_30s_buffers[session_id] = []
            _session_last_30s_save[session_id] = time.time()

        # Append new incoming raw samples
        _session_live_buffers[session_id].extend(buffer)
        _session_30s_buffers[session_id].extend(buffer)

        # 1. LIVE DATA (every single row saved to eeg_live_data for smooth graph)
        live_rows, remaining_tail = compute_sliding_live_metrics(
            _session_live_buffers[session_id],
            window_size=128,
            step_size=32,
            sample_rate=256,
        )
        _session_live_buffers[session_id] = remaining_tail

        # If buffer is still building up to window_size, insert a single point once >= 16 samples
        if not live_rows and len(_session_live_buffers[session_id]) >= 16:
            single = analyze_eeg_buffer(_session_live_buffers[session_id])
            live_rows = [{
                "alpha": single["alpha"],
                "beta": single["beta"],
                "theta": single["theta"],
                "ratio": single["ratio"],
                "mind_state": single["mind_state"],
                "timestamp": datetime.datetime.now(),
            }]

        if live_rows:
            save_live_rows_to_db(live_rows, session_id=session_id)

        # 2. 30-SECOND EMOTION (saved to eeg_state ONLY every 30 seconds)
        now = time.time()
        elapsed = now - _session_last_30s_save[session_id]
        saved_30s_state = None

        if elapsed >= 30.0 or len(_session_30s_buffers[session_id]) >= 256 * 30:
            saved_30s_state = analyze_30s_data(_session_30s_buffers[session_id], sample_rate=256)
            save_30s_emotion_to_db(saved_30s_state, session_id=session_id)
            _session_last_30s_save[session_id] = now
            # Retain the last 2 seconds to keep stream context continuous
            _session_30s_buffers[session_id] = _session_30s_buffers[session_id][-512:]

    return {
        "status": "ok",
        "live_rows_inserted": len(live_rows),
        "saved_30s_emotion": saved_30s_state,
    }


def run_simulated_stream(session_id: int | None = None):
    """Generates continuous realistic 4-channel brainwaves and streams them directly into the DB pipeline."""
    print("\n" + "=" * 60)
    print("🧪 STARTING LIVE SIMULATED EEG STREAM (Theta, Alpha, Beta)")
    print("   -> Writing continuous rows to eeg_live_data (for smooth graph)")
    print("   -> Evaluating 30-second emotion to eeg_state")
    print("   -> Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    sample_rate = 256
    chunk_samples = 32  # Send 32 samples every 0.125s
    dt = chunk_samples / sample_rate
    t = 0.0

    try:
        while True:
            time_axis = np.linspace(t, t + dt, chunk_samples, endpoint=False)
            t += dt

            # Synthetic brainwaves in microvolts
            alpha_wave = 15.0 * np.sin(2 * np.pi * 10 * time_axis)
            beta_wave = 5.0 * np.sin(2 * np.pi * 20 * time_axis)
            theta_wave = 7.0 * np.sin(2 * np.pi * 6 * time_axis)
            noise = np.random.randn(chunk_samples) * 2.0

            tp9 = alpha_wave + noise
            af7 = beta_wave + noise
            af8 = beta_wave + noise
            tp10 = theta_wave + noise
            chunk = np.column_stack([tp9, af7, af8, tp10]).tolist()

            res = process_and_save_eeg_stream(chunk, session_id=session_id)
            if res.get("saved_30s_emotion"):
                print(f"🎉 30-Second Emotion committed to eeg_state: {res['saved_30s_emotion']}")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n⏹️ Simulated stream stopped.")


def main():
    if "--simulate" in sys.argv or "--test" in sys.argv:
        run_simulated_stream()
        return

    print("Looking for Muse EEG stream... (Make sure 'muselsl stream' is running)")
    try:
        from pylsl import StreamInlet, resolve_stream
    except ImportError:
        print("❌ pylsl is not installed. Please run: pip install pylsl (or run with --simulate)")
        return

    # Check for active stream with 3-second timeout instead of blocking forever
    try:
        streams = resolve_stream("type", "EEG", timeout=3.0)
    except Exception as e:
        print(f"⚠️ LSL stream lookup error: {e}")
        streams = []

    if not streams:
        print("\n⚠️ No active LSL EEG stream found from Muse headband after 3 seconds.")
        print("💡 If your Muse is connected to your iPhone app:")
        print("   -> Keep FastAPI running ('python FYP2/main.py'). The iOS app streams directly to the database!")
        print("   -> You do NOT need to run this script when using the iPhone app.")
        print("💡 If you want to stream from your Mac's Bluetooth:")
        print("   -> Run 'muselsl stream' in another terminal first.")
        print("💡 If you want to test with synthetic brainwaves:")
        print("   -> Run: python FYP2/EEGHANDLING/eegstream.py --simulate\n")
        return

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    channel_names = get_channel_names(inlet)

    print("Connected to Muse 2! Starting continuous stream...")
    print(f"Stream Name: {info.name()} | Rate: {info.nominal_srate()} Hz | Nodes ({len(channel_names)}): {', '.join(channel_names)}\n")

    live_buf = []
    buf_30s = []
    last_30s_time = time.time()
    last_display_time = 0

    while True:
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is None:
            sys.stdout.write("\r⚠️  Waiting for EEG data stream from Muse headband...")
            sys.stdout.flush()
            time.sleep(0.1)
            continue

        live_buf.append(sample)
        buf_30s.append(sample)
        current_time = time.time()
        elapsed = current_time - last_30s_time

        # Every ~32 samples (0.125s), compute and save live data row
        if len(live_buf) >= 128:
            rows, live_buf = compute_sliding_live_metrics(live_buf, window_size=128, step_size=32)
            if rows:
                save_live_rows_to_db(rows, session_id=None)

        # Display status in console every 0.25 seconds
        if current_time - last_display_time >= 0.25:
            last_display_time = current_time
            signals_str = " | ".join(
                f"{name}: {val:+.1f}µV" for name, val in zip(channel_names, sample)
            )
            live_state = analyze_30s_data(buf_30s[-1280:]) if len(buf_30s) >= 256 * 3 else "Initializing..."
            sys.stdout.write(
                f"\r📡 [{elapsed:4.1f}s / 30.0s] Signals: {signals_str} | Live: {live_state}    "
            )
            sys.stdout.flush()

        # Save 30-second emotion to eeg_state every 30 seconds
        if elapsed >= 30.0 or len(buf_30s) >= 256 * 30:
            sys.stdout.write("\n")
            sys.stdout.flush()
            current_state = analyze_30s_data(buf_30s)
            save_30s_emotion_to_db(current_state, session_id=None)
            last_30s_time = time.time()
            buf_30s = buf_30s[-512:]


if __name__ == "__main__":
    main()
