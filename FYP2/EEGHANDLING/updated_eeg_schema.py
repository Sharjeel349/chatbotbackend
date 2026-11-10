#!/usr/bin/env python3
"""
========================================================================================
UPDATED EEG SCHEMA PROCESSOR (Alpha, Beta, Theta -> 6 Cognitive/Stress States)
========================================================================================

Brain State Mapping:
    1 => Neutral           (Balanced resting baseline)
    2 => Calm              (Alpha dominant, relaxed, peaceful)
    3 => Focused           (Beta engagement elevated, active cognitive processing)
    4 => Mild Stress       (Early beta rise above alpha, onset of tension)
    5 => Moderate Stress   (Significant beta elevation, active stress response)
    6 => Severe Stress     (Extreme beta surge, severe anxiety / mental overload)

Supported Inputs:
    1. Live LSL stream from Mac (Muse 2 headband via `muselsl stream`)
    2. Simulated stream (`--simulate`) for offline testing / committee demonstration
    3. Importable function/class for FastAPI routes & background services

Bands Monitored:
    - Theta (θ): 4.0 - 8.0 Hz   (Drowsiness, meditation, subconscious processing)
    - Alpha (α): 8.0 - 13.0 Hz  (Relaxation, calm alertness, idle state)
    - Beta  (β): 13.0 - 30.0 Hz (Active thinking, problem solving, anxiety, stress)
========================================================================================
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from FYP2.core.config import DB_CONFIG

# Thread-safe in-memory stream buffers for live sessions (supporting continuous frontend stream)
_stream_lock = threading.Lock()
_session_live_buffers: dict[Any, list[list[float]]] = {}
_session_30s_buffers: dict[Any, list[list[float]]] = {}
_session_last_30s_save: dict[Any, float] = {}

# --------------------------------------------------------------------------------------
# 6 Cognitive States Definition & Thresholds
# --------------------------------------------------------------------------------------
STATE_NEUTRAL = 1
STATE_CALM = 2
STATE_FOCUSED = 3
STATE_MILD_STRESS = 4
STATE_MODERATE_STRESS = 5
STATE_SEVERE_STRESS = 6

STATE_NAMES: dict[int, str] = {
    STATE_NEUTRAL: "Neutral",
    STATE_CALM: "Calm",
    STATE_FOCUSED: "Focused",
    STATE_MILD_STRESS: "Mild Stress",
    STATE_MODERATE_STRESS: "Moderate Stress",
    STATE_SEVERE_STRESS: "Severe Stress",
}

# ANSI terminal colors for live display
STATE_COLORS: dict[int, str] = {
    STATE_NEUTRAL: "\033[94m",          # Blue
    STATE_CALM: "\033[92m",             # Green
    STATE_FOCUSED: "\033[96m",          # Cyan
    STATE_MILD_STRESS: "\033[93m",      # Yellow
    STATE_MODERATE_STRESS: "\033[38;5;208m", # Orange
    STATE_SEVERE_STRESS: "\033[91;1m",   # Bright Bold Red
}
COLOR_RESET = "\033[0m"

# Classification Thresholds (Beta/Alpha Ratio & Engagement Index)
STRESS_SEVERE_BAR = 2.8      # Beta / Alpha >= 2.8 => Severe Stress (State 6)
STRESS_MODERATE_BAR = 2.0    # Beta / Alpha >= 2.0 => Moderate Stress (State 5)
STRESS_MILD_BAR = 1.5        # Beta / Alpha >= 1.5 => Mild Stress (State 4)
CALM_BAR = 0.75              # Beta / Alpha < 0.75 => Calm (State 2)
FOCUS_EAR = 0.45             # Beta / (Alpha + Theta) >= 0.45 with moderate BAR => Focused (State 3)


@dataclass
class EEGAnalysisResult:
    state_code: int
    state_name: str
    theta: float
    alpha: float
    beta: float
    beta_alpha_ratio: float
    engagement_ratio: float
    timestamp: datetime.datetime
    raw_channels: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_code": self.state_code,
            "state_name": self.state_name,
            "theta": round(self.theta, 4),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "beta_alpha_ratio": round(self.beta_alpha_ratio, 4),
            "engagement_ratio": round(self.engagement_ratio, 4),
            "timestamp": self.timestamp.isoformat(),
            "raw_channels": self.raw_channels or {},
        }


def classify_brain_state(alpha: float, beta: float, theta: float) -> tuple[int, str, float, float]:
    """
    Evaluates Alpha, Beta, and Theta powers and returns:
        (state_code, state_name, beta_alpha_ratio, engagement_ratio)

    States:
        1 => Neutral
        2 => Calm
        3 => Focused
        4 => Mild Stress
        5 => Moderate Stress
        6 => Severe Stress
    """
    # Protect against division by zero
    safe_alpha = max(alpha, 1e-9)
    safe_sum = max(alpha + theta, 1e-9)

    bar = float(beta / safe_alpha)              # Beta / Alpha Ratio
    ear = float(beta / safe_sum)                # Engagement Ratio: Beta / (Alpha + Theta)
    total_power = max(alpha + beta + theta, 1e-9)
    alpha_prop = float(alpha / total_power)     # Proportion of Alpha

    # 1. Stress Hierarchy (Severe -> Moderate -> Mild)
    if bar >= STRESS_SEVERE_BAR:
        return STATE_SEVERE_STRESS, STATE_NAMES[STATE_SEVERE_STRESS], bar, ear

    if bar >= STRESS_MODERATE_BAR:
        return STATE_MODERATE_STRESS, STATE_NAMES[STATE_MODERATE_STRESS], bar, ear

    if bar >= STRESS_MILD_BAR:
        return STATE_MILD_STRESS, STATE_NAMES[STATE_MILD_STRESS], bar, ear

    # 2. Calm State (Dominant Alpha, Low Beta)
    if bar < CALM_BAR or (alpha_prop >= 0.45 and bar < 0.90):
        return STATE_CALM, STATE_NAMES[STATE_CALM], bar, ear

    # 3. Focused State (Active Beta, Controlled Theta, High Engagement, No Stress)
    if ear >= FOCUS_EAR and theta <= beta * 1.1 and bar >= 1.0:
        return STATE_FOCUSED, STATE_NAMES[STATE_FOCUSED], bar, ear

    # 4. Neutral Baseline (Balanced activity)
    return STATE_NEUTRAL, STATE_NAMES[STATE_NEUTRAL], bar, ear


def compute_spectral_powers(
    buffer: list[list[float]] | np.ndarray,
    sample_rate: int = 256,
) -> tuple[float, float, float]:
    """
    Computes Theta (4-8 Hz), Alpha (8-13 Hz), and Beta (13-30 Hz) powers
    from raw multi-channel EEG samples using FFT with a Hanning window.
    """
    data = np.asarray(buffer, dtype=float)
    if data.ndim != 2 or data.shape[0] < 16:
        return 0.0, 0.0, 0.0

    # Muse standard has 4 main channels: TP9, AF7, AF8, TP10
    channels_to_use = min(5, data.shape[1])
    eeg = data[:, :channels_to_use].copy()

    # De-mean per channel
    eeg -= np.mean(eeg, axis=0, keepdims=True)

    # Apply Hanning window across samples to reduce spectral leakage
    window = np.hanning(eeg.shape[0])[:, None]
    eeg_windowed = eeg * window

    # Fast Fourier Transform (real-input)
    fft_vals = np.fft.rfft(eeg_windowed, axis=0)
    power_spectrum = np.mean(np.abs(fft_vals) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(eeg.shape[0], d=1.0 / sample_rate)

    def band_sum(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power_spectrum[mask]))

    theta = band_sum(4.0, 8.0)
    alpha = band_sum(8.0, 13.0)
    beta = band_sum(13.0, 30.0)

    return theta, alpha, beta


def analyze_eeg_buffer(
    buffer: list[list[float]] | np.ndarray,
    sample_rate: int = 256,
    channel_names: list[str] | None = None,
) -> EEGAnalysisResult:
    """
    End-to-end analysis of an EEG window:
    Extracts Theta, Alpha, Beta powers and assigns one of the 6 cognitive states.
    """
    now = datetime.datetime.now()
    theta, alpha, beta = compute_spectral_powers(buffer, sample_rate=sample_rate)
    code, name, bar, ear = classify_brain_state(alpha, beta, theta)

    # Compute mean raw microvolts for display
    raw_channels: dict[str, float] = {}
    if buffer is not None and len(buffer) > 0:
        data = np.asarray(buffer, dtype=float)
        if data.ndim == 2:
            means = np.mean(data[-16:], axis=0)  # average of recent samples
            names = channel_names or ["TP9", "AF7", "AF8", "TP10", "AUX"]
            for idx, m in enumerate(means):
                lbl = names[idx] if idx < len(names) else f"CH{idx+1}"
                raw_channels[lbl] = round(float(m), 2)

    return EEGAnalysisResult(
        state_code=code,
        state_name=name,
        theta=theta,
        alpha=alpha,
        beta=beta,
        beta_alpha_ratio=bar,
        engagement_ratio=ear,
        timestamp=now,
        raw_channels=raw_channels,
    )


# --------------------------------------------------------------------------------------
# Frontend Stream Ingestion & Continuous DB Storage (eeg_live_data + eeg_state)
# --------------------------------------------------------------------------------------
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
    except Exception as e:
        print(f"❌ DATABASE ERROR (eeg_live_data): {e}")


def compute_sliding_live_metrics(
    buffer: list[list[float]],
    window_size: int = 128,
    step_size: int = 32,
    sample_rate: int = 256,
) -> tuple[list[dict], list[list[float]]]:
    """
    Computes smooth, overlapping spectral metrics (Theta, Alpha, Beta, Ratio)
    using the updated 6-state schema for every step across the buffer so eeg_live_data
    produces a continuous smooth graph for the frontend.
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

        def band_power(low: float, high: float) -> float:
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                return 0.0
            return float(np.sum(power[mask]))

        theta = band_power(4.0, 8.0)
        alpha = band_power(8.0, 13.0)
        beta = band_power(13.0, 30.0)

        code, state_name, bar, ear = classify_brain_state(alpha, beta, theta)

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
            "ratio": round(bar, 4),
            "mind_state": state_name,
            "tp9": round(tp9, 2) if tp9 is not None else None,
            "af7": round(af7, 2) if af7 is not None else None,
            "af8": round(af8, 2) if af8 is not None else None,
            "tp10": round(tp10, 2) if tp10 is not None else None,
            "timestamp": row_time,
        })

        idx += step_size

    remaining_tail = eeg_all[idx:].tolist()
    return live_rows, remaining_tail


def process_and_save_eeg_stream(buffer: list, session_id: int | None = None) -> dict:
    """
    Primary ingestion entrypoint for the FastAPI EEG stream route (/api/eeg/stream):
    1. LIVE DATA: Inserts every single row of computed wave metrics into eeg_live_data
       using an overlapping sliding window so the graph will be dense and smooth.
    2. 30-SECOND EMOTION: Evaluates the full 30s epoch and saves into eeg_state ONLY
       every 30 seconds using the updated 6-state schema.
    """
    if not buffer:
        return {"status": "empty"}

    with _stream_lock:
        if session_id not in _session_live_buffers:
            _session_live_buffers[session_id] = []
            _session_30s_buffers[session_id] = []
            _session_last_30s_save[session_id] = time.time()

        # Append new incoming raw samples from frontend
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
                "alpha": round(single.alpha, 4),
                "beta": round(single.beta, 4),
                "theta": round(single.theta, 4),
                "ratio": round(single.beta_alpha_ratio, 4),
                "mind_state": single.state_name,
                "timestamp": datetime.datetime.now(),
            }]

        if live_rows:
            save_live_rows_to_db(live_rows, session_id=session_id)

        # 2. 30-SECOND EMOTION (saved to eeg_state ONLY every 30 seconds)
        now = time.time()
        elapsed = now - _session_last_30s_save[session_id]
        saved_30s_state = None

        if elapsed >= 30.0 or len(_session_30s_buffers[session_id]) >= 256 * 30:
            final_res = analyze_eeg_buffer(_session_30s_buffers[session_id], sample_rate=256)
            saved_30s_state = final_res.state_name
            save_eeg_state_to_db(final_res, session_id=session_id)
            _session_last_30s_save[session_id] = now
            # Retain the last 2 seconds to keep stream context continuous
            _session_30s_buffers[session_id] = _session_30s_buffers[session_id][-512:]
            print("\n" + "=" * 65)
            print(f"🧠 [30s EEG_State SAVED]: State {final_res.state_code} => {saved_30s_state} (session_id={session_id})")
            print("=" * 65 + "\n")

    return {
        "status": "ok",
        "live_rows_inserted": len(live_rows),
        "saved_30s_emotion": saved_30s_state,
    }


# --------------------------------------------------------------------------------------
# CLI Display Formatter
# --------------------------------------------------------------------------------------
def format_bar_gauge(val: float, max_val: float = 30.0, bar_len: int = 10) -> str:
    """Generates a small Unicode bar for terminal visualization."""
    filled = int(min(max(val / max_val, 0.0), 1.0) * bar_len)
    return "█" * filled + "░" * (bar_len - filled)


def print_collection_progress(
    elapsed: float,
    target: float = 30.0,
    raw_channels: dict[str, float] | None = None,
) -> None:
    """Updates a single in-place countdown line during the 30-second recording window."""
    pct = min(elapsed / target, 1.0)
    bar_len = 20
    filled = int(pct * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    ch_str = ""
    if raw_channels:
        ch_str = " | " + " ".join(
            f"{k}:{v:+.0f}µV" for k, v in raw_channels.items() if k in ["TP9", "AF7", "AF8", "TP10"]
        )

    sys.stdout.write(
        f"\r⏳ [Recording 30s EEG Window] [{bar}] {elapsed:4.1f}s / {target:4.1f}s ({pct*100:3.0f}%){ch_str}   "
    )
    sys.stdout.flush()


def print_30s_summary(
    result: EEGAnalysisResult,
    epoch_idx: int,
    start_sec: float,
    end_sec: float,
    distribution: dict[str, float] | None = None,
) -> None:
    """Prints a clean, comprehensive 30-second aggregated emotion summary."""
    color = STATE_COLORS.get(result.state_code, "\033[0m")
    border = "=" * 80
    sub_border = "-" * 80

    # Wipe the in-place progress line cleanly
    sys.stdout.write("\r" + " " * 95 + "\r")
    sys.stdout.flush()

    print("\n" + border)
    print(f"🧠 30-SECOND EMOTION SUMMARY [Epoch #{epoch_idx} | {start_sec:05.1f}s -> {end_sec:05.1f}s]")
    print(border)
    print(f"  ▶ FINAL EMOTION STATE  : {color}[State {result.state_code} => {result.state_name.upper()}]{COLOR_RESET}")
    print(sub_border)
    print(f"  • Alpha Power (α)      : {result.alpha:,.1f}  (Relaxation / Calm baseline)")
    print(f"  • Beta Power  (β)      : {result.beta:,.1f}  (Active thinking / Tension)")
    print(f"  • Theta Power (θ)      : {result.theta:,.1f}  (Subconscious / Idle / Drowsiness)")
    print(f"  • Beta/Alpha Ratio(BAR): {result.beta_alpha_ratio:4.2f}")
    print(f"  • Cognitive Focus (EAR): {result.engagement_ratio:4.2f}")

    if distribution:
        dist_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        dist_str = " | ".join(f"{st}: {pct:.0f}%" for st, pct in dist_items if pct > 0)
        print(f"  • 30s State Breakdown  : {dist_str}")

    if result.raw_channels:
        ch_str = " | ".join(f"{k}: {v:+.1f}µV" for k, v in result.raw_channels.items())
        print(f"  • Mean Electrode Offset: {ch_str}")

    print(border + "\n")
    sys.stdout.flush()


def print_state_banner(result: EEGAnalysisResult, elapsed: float | None = None) -> None:
    color = STATE_COLORS.get(result.state_code, "\033[0m")
    code = result.state_code
    name = result.state_name

    # Mini bar graphs
    th_bar = format_bar_gauge(result.theta, 40.0, 8)
    al_bar = format_bar_gauge(result.alpha, 40.0, 8)
    be_bar = format_bar_gauge(result.beta, 40.0, 8)

    channels_str = ""
    if result.raw_channels:
        channels_str = " | ".join(f"{k}:{v:+.1f}µV" for k, v in result.raw_channels.items() if k in ["TP9", "AF7", "AF8", "TP10"])

    time_tag = f"[{elapsed:5.1f}s]" if elapsed is not None else ""

    sys.stdout.write(
        f"\r{time_tag} {color}▶ [State {code} => {name:<15}]{COLOR_RESET} "
        f"θ:[{th_bar}] {result.theta:5.1f} | "
        f"α:[{al_bar}] {result.alpha:5.1f} | "
        f"β:[{be_bar}] {result.beta:5.1f} | "
        f"BAR:{result.beta_alpha_ratio:4.2f} "
        f"({channels_str})   "
    )
    sys.stdout.flush()


# --------------------------------------------------------------------------------------
# Mac LSL Stream Receiver (30-Second Aggregate Epochs)
# --------------------------------------------------------------------------------------
def run_mac_lsl_stream(
    sample_rate: int = 256,
    window_seconds: float = 30.0,
    live_subseconds: bool = False,
    save_db: bool = False,
    session_id: int | None = None,
):
    """
    Connects to the Lab Streaming Layer (LSL) stream on Mac (from `muselsl stream`).
    Records 30 seconds of brainwaves into a full epoch buffer, computes the overall
    spectral powers & state distribution, and prints ONE stable emotion at the end of 30s.
    """
    print("\n" + "=" * 80)
    print("🧠 MAC EEG STREAM PROCESSOR (30-Second Emotion Aggregator)")
    print("=" * 80)
    print("States: 1=>Neutral | 2=>Calm | 3=>Focused | 4=>Mild Stress | 5=>Mod Stress | 6=>Severe Stress")
    print(f"Aggregation Window: {window_seconds:.0f} seconds (no per-second spam)")
    print("Looking for Muse EEG stream on Mac (make sure 'muselsl stream' is running)...")

    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError:
        print("\n❌ 'pylsl' package not found.")
        print("   Install it using: pip install pylsl")
        print("   Or run in simulation mode: python updated_eeg_schema.py --simulate")
        return

    try:
        streams = resolve_byprop("type", "EEG", timeout=3.0)
    except Exception as e:
        print(f"⚠️ LSL search error: {e}")
        streams = []

    if not streams:
        print("\n⚠️ No active LSL EEG stream found on your Mac within 3 seconds.")
        print("💡 How to connect your Muse 2 on Mac:")
        print("   1. Turn on Muse headband (LEDs bouncing).")
        print("   2. Open another terminal tab and run: muselsl stream")
        print("   3. Re-run this script: python FYP2/EEGHANDLING/updated_eeg_schema.py")
        print("💡 Or to test immediately without headset hardware:")
        print("   Run: python FYP2/EEGHANDLING/updated_eeg_schema.py --simulate\n")
        return

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    nominal_rate = int(info.nominal_srate()) or sample_rate

    # Discover channel names
    ch_names = []
    try:
        ch = info.desc().child("channels").child("channel")
        while not ch.empty():
            lbl = ch.child_value("label")
            if lbl:
                ch_names.append(lbl)
            ch = ch.next_sibling("channel")
    except Exception:
        pass
    if not ch_names:
        ch_names = ["TP9", "AF7", "AF8", "TP10", "AUX"]

    print(f"✅ Connected to Mac Muse Stream: {info.name()} ({nominal_rate} Hz)")
    print(f"   Channels: {', '.join(ch_names)}")
    print(f"   Recording 30s epochs... Press Ctrl+C to stop.\n")

    epoch_samples = int(nominal_rate * window_seconds)
    epoch_buffer: list[list[float]] = []
    subwindow_states: list[str] = []

    epoch_start_time = time.time()
    total_start_time = epoch_start_time
    last_subwindow_eval = epoch_start_time
    last_progress_display = 0.0
    epoch_idx = 1

    try:
        while True:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                sys.stdout.write("\r⏳ Waiting for incoming EEG samples from Muse headband...")
                sys.stdout.flush()
                time.sleep(0.05)
                continue

            epoch_buffer.append(sample)
            now = time.time()
            elapsed_epoch = now - epoch_start_time

            # Compute short 1-second state quietly in the background to build the 30s distribution
            if len(epoch_buffer) >= nominal_rate and (now - last_subwindow_eval) >= 1.0:
                last_subwindow_eval = now
                sub_res = analyze_eeg_buffer(epoch_buffer[-nominal_rate:], sample_rate=nominal_rate, channel_names=ch_names)
                subwindow_states.append(sub_res.state_name)

            # In live mode (if requested), print banners; otherwise update single progress line
            if live_subseconds:
                if (now - last_progress_display) >= 0.25 and len(epoch_buffer) >= 32:
                    last_progress_display = now
                    live_res = analyze_eeg_buffer(epoch_buffer[-128:], sample_rate=nominal_rate, channel_names=ch_names)
                    print_state_banner(live_res, elapsed=elapsed_epoch)
            else:
                if (now - last_progress_display) >= 0.20:
                    last_progress_display = now
                    # Extract last channel readings for signal quality display
                    raw_dict = {}
                    if epoch_buffer:
                        last_s = epoch_buffer[-1]
                        for idx, val in enumerate(last_s[: len(ch_names)]):
                            raw_dict[ch_names[idx]] = round(float(val), 1)
                    print_collection_progress(elapsed_epoch, target=window_seconds, raw_channels=raw_dict)

            # When 30 seconds have elapsed (or buffer reached full epoch size)
            if elapsed_epoch >= window_seconds or len(epoch_buffer) >= epoch_samples:
                # 1. Full 30-second Spectral Analysis
                final_res = analyze_eeg_buffer(epoch_buffer, sample_rate=nominal_rate, channel_names=ch_names)

                # 2. Compute 30s distribution breakdown
                dist_pct: dict[str, float] = {}
                if subwindow_states:
                    total_subs = len(subwindow_states)
                    for st_name in STATE_NAMES.values():
                        cnt = subwindow_states.count(st_name)
                        if cnt > 0:
                            dist_pct[st_name] = round((cnt / total_subs) * 100.0, 1)

                # 3. Print the 30s Emotion Summary Card
                epoch_end_time = now
                start_offset = epoch_start_time - total_start_time
                end_offset = epoch_end_time - total_start_time
                print_30s_summary(final_res, epoch_idx, start_offset, end_offset, distribution=dist_pct)

                # 4. Save to DB if requested
                if save_db:
                    save_eeg_state_to_db(final_res, session_id=session_id)

                # 5. Reset for the next 30-second epoch
                epoch_idx += 1
                epoch_start_time = time.time()
                last_subwindow_eval = epoch_start_time
                epoch_buffer = []
                subwindow_states = []

    except KeyboardInterrupt:
        print(f"\n{COLOR_RESET}⏹️ EEG recording stopped by user.")


# --------------------------------------------------------------------------------------
# Simulation Mode (30-Second Aggregate Epochs)
# --------------------------------------------------------------------------------------
def run_simulation(
    window_seconds: float = 30.0,
    live_subseconds: bool = False,
    save_db: bool = False,
    session_id: int | None = None,
):
    """
    Generates synthetic 4-channel brainwaves that dynamically cycle through
    all 6 states across 30-second epochs:
        Epoch 1 (00-30s)  => State 1: Neutral
        Epoch 2 (30-60s)  => State 2: Calm
        Epoch 3 (60-90s)  => State 3: Focused
        Epoch 4 (90-120s) => State 4: Mild Stress
        Epoch 5 (120-150s)=> State 5: Moderate Stress
        Epoch 6 (150-180s)=> State 6: Severe Stress
    """
    print("\n" + "=" * 80)
    print(f"🧪 SIMULATED EEG MAC STREAM ({window_seconds:.0f}-Second Epoch Aggregator)")
    print("=" * 80)
    print("Simulated 30s Cycles:")
    print("  Epoch 1 => 1: Neutral")
    print("  Epoch 2 => 2: Calm")
    print("  Epoch 3 => 3: Focused")
    print("  Epoch 4 => 4: Mild Stress")
    print("  Epoch 5 => 5: Moderate Stress")
    print("  Epoch 6 => 6: Severe Stress")
    print("  (Repeats cycle continuously. Press Ctrl+C to stop)\n")

    sample_rate = 256
    chunk_samples = 32
    dt = chunk_samples / sample_rate
    t = 0.0

    epoch_samples = int(sample_rate * window_seconds)
    epoch_buffer: list[list[float]] = []
    subwindow_states: list[str] = []

    epoch_start_time = time.time()
    total_start_time = epoch_start_time
    last_subwindow_eval = epoch_start_time
    last_progress_display = 0.0
    epoch_idx = 1

    try:
        while True:
            # Determine which state phase we are in (cycles every 6 epochs)
            phase = (epoch_idx - 1) % 6
            if phase == 0:
                # 1 => Neutral
                a_amp, b_amp, th_amp = 11.0, 10.5, 15.0
            elif phase == 1:
                # 2 => Calm
                a_amp, b_amp, th_amp = 22.0, 5.0, 8.0
            elif phase == 2:
                # 3 => Focused
                a_amp, b_amp, th_amp = 11.0, 11.5, 4.0
            elif phase == 3:
                # 4 => Mild Stress
                a_amp, b_amp, th_amp = 10.0, 13.0, 8.0
            elif phase == 4:
                # 5 => Moderate Stress
                a_amp, b_amp, th_amp = 9.0, 14.5, 7.0
            else:
                # 6 => Severe Stress
                a_amp, b_amp, th_amp = 7.0, 16.0, 5.0

            time_axis = np.linspace(t, t + dt, chunk_samples, endpoint=False)
            t += dt

            # Generate synthetic wave frequencies
            alpha_signal = a_amp * np.sin(2 * np.pi * 10.0 * time_axis)
            beta_signal = b_amp * np.sin(2 * np.pi * 20.0 * time_axis)
            theta_signal = th_amp * np.sin(2 * np.pi * 6.0 * time_axis)
            noise = np.random.randn(chunk_samples) * 0.8

            # Realistic 4-channel volume conduction mix across Muse headband
            tp9 = alpha_signal * 0.7 + beta_signal * 0.5 + theta_signal * 0.5 + noise
            af7 = alpha_signal * 0.5 + beta_signal * 0.7 + theta_signal * 0.4 + noise
            af8 = alpha_signal * 0.5 + beta_signal * 0.7 + theta_signal * 0.4 + noise
            tp10 = alpha_signal * 0.7 + beta_signal * 0.5 + theta_signal * 0.5 + noise

            chunk = np.column_stack([tp9, af7, af8, tp10]).tolist()
            epoch_buffer.extend(chunk)

            now = time.time()
            elapsed_epoch = now - epoch_start_time

            # Compute short 1-second state quietly to build distribution
            if len(epoch_buffer) >= sample_rate and (now - last_subwindow_eval) >= 1.0:
                last_subwindow_eval = now
                sub_res = analyze_eeg_buffer(epoch_buffer[-sample_rate:], sample_rate=sample_rate)
                subwindow_states.append(sub_res.state_name)

            if live_subseconds:
                if (now - last_progress_display) >= 0.25 and len(epoch_buffer) >= 64:
                    last_progress_display = now
                    live_res = analyze_eeg_buffer(epoch_buffer[-128:], sample_rate=sample_rate)
                    print_state_banner(live_res, elapsed=elapsed_epoch)
            else:
                if (now - last_progress_display) >= 0.20:
                    last_progress_display = now
                    raw_dict = {
                        "TP9": round(float(tp9[-1]), 1),
                        "AF7": round(float(af7[-1]), 1),
                        "AF8": round(float(af8[-1]), 1),
                        "TP10": round(float(tp10[-1]), 1),
                    }
                    print_collection_progress(elapsed_epoch, target=window_seconds, raw_channels=raw_dict)

            # When 30 seconds have elapsed
            if elapsed_epoch >= window_seconds or len(epoch_buffer) >= epoch_samples:
                final_res = analyze_eeg_buffer(epoch_buffer, sample_rate=sample_rate)

                dist_pct: dict[str, float] = {}
                if subwindow_states:
                    total_subs = len(subwindow_states)
                    for st_name in STATE_NAMES.values():
                        cnt = subwindow_states.count(st_name)
                        if cnt > 0:
                            dist_pct[st_name] = round((cnt / total_subs) * 100.0, 1)

                epoch_end_time = now
                start_offset = epoch_start_time - total_start_time
                end_offset = epoch_end_time - total_start_time
                print_30s_summary(final_res, epoch_idx, start_offset, end_offset, distribution=dist_pct)

                if save_db:
                    save_eeg_state_to_db(final_res, session_id=session_id)

                epoch_idx += 1
                epoch_start_time = time.time()
                last_subwindow_eval = epoch_start_time
                epoch_buffer = []
                subwindow_states = []

            time.sleep(dt)

    except KeyboardInterrupt:
        print(f"\n{COLOR_RESET}⏹️ Simulation stopped.")


# --------------------------------------------------------------------------------------
# Optional PostgreSQL Database Persistence
# --------------------------------------------------------------------------------------
def save_eeg_state_to_db(result: EEGAnalysisResult, session_id: int | None = None) -> bool:
    """
    Saves the analyzed state into PostgreSQL 'eeg_state' table:
        (mind_state, session_id, timestamp)
    """
    try:
        from FYP2.core.config import DB_CONFIG
        import psycopg2

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO eeg_state (mind_state, session_id) VALUES (%s, %s);"
        state_str = result.state_name
        cursor.execute(query, (state_str, session_id))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"💾 [eeg_state] Saved 30s Emotion: {state_str} (session_id={session_id})")
        return True
    except Exception as e:
        print(f"\n⚠️ Database insert error (eeg_state): {e}")
        return False


# --------------------------------------------------------------------------------------
# Main Execution Entrypoint
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Updated EEG Schema (30-Second Emotion Aggregator)")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulated brainwaves cycling through all 6 states without hardware",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Aggregation window in seconds (default: 30.0s)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show continuous sub-second updates instead of 30s aggregate summary",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=256,
        help="Sampling rate in Hz (default: 256)",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Also log states to PostgreSQL 'eeg_state' table",
    )
    args = parser.parse_args()

    if args.simulate:
        run_simulation(
            window_seconds=args.window,
            live_subseconds=args.live,
            save_db=args.save_db,
        )
    else:
        run_mac_lsl_stream(
            sample_rate=args.rate,
            window_seconds=args.window,
            live_subseconds=args.live,
            save_db=args.save_db,
        )


if __name__ == "__main__":
    main()
