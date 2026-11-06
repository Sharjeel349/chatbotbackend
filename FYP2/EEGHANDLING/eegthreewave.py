"""
EEG Three-Wave Cognitive & Emotion Streamer (eegthreewave.py)
------------------------------------------------------------
Analyzes Theta (4-8 Hz), Alpha (8-13 Hz), and Beta (13-30 Hz) bands from
Muse 2 EEG headband (TP9, AF7, AF8, TP10) via Lab Streaming Layer (LSL).

Computes:
  1. Band Powers: Theta, Alpha, Beta (via FFT Power Spectral Density)
  2. Continuous Stress Level (0 - 100%)
  3. Continuous Focus / Engagement Score (0 - 100% via NASA Pope Index)
  4. Cognitive Fatigue & Confusion Indices
  5. Frontal Alpha Asymmetry (FAA between AF7 and AF8)
  6. Classified Cognitive Emotions:
     - STRESSED   (High Beta, Low Alpha)
     - FOCUSED    (High Beta, Low Theta)
     - CONFUSED   (High Theta surge, Suppressed Alpha)
     - CALM       (Dominant Alpha, Low Beta/Theta)
     - FATIGUED   (High Theta, Low Beta/Alpha)
     - NEUTRAL    (Balanced Baseline)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FYP2.core.config import DB_CONFIG

try:
    from pylsl import StreamInlet, resolve_stream
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False


DEFAULT_MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]


# ==============================================================================
# 1. CHANNEL DISCOVERY & CONTACT QUALITY CHECK
# ==============================================================================

def get_channel_names(inlet) -> List[str]:
    """Retrieve electrode labels from the incoming LSL stream metadata."""
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
            ch_names = DEFAULT_MUSE_CHANNELS[:count]
        else:
            ch_names = [f"CH{i+1}" for i in range(count)]

    return ch_names


def verify_electrode_contact(raw_eeg_window: np.ndarray) -> Tuple[bool, str]:
    """
    Checks if electrodes are making proper skin contact on the forehead/ears.
    - Off-head / Railed: Signal saturates (> 450 uV) or produces extreme standard deviation.
    - Disconnected / Flatline: Near-zero variance (< 1.5 uV) or NaN values.
    """
    if raw_eeg_window.size == 0 or np.any(np.isnan(raw_eeg_window)):
        return False, "DISCONNECTED"

    stds = np.std(raw_eeg_window, axis=0)

    # Any channel with standard deviation < 1.5 uV is flatlined
    if np.any(stds < 1.5):
        return False, "FLATLINE_DETECTED"

    # Any channel with standard deviation > 450 uV is loose/noisy
    if np.any(stds > 450.0):
        return False, "POOR_CONTACT_NOISE"

    return True, "GOOD"


# ==============================================================================
# 2. THREE-WAVE POWER SPECTRAL DENSITY & RATIOS
# ==============================================================================

def extract_three_waves(
    buffer: List[List[float]],
    sample_rate: int = 256,
) -> Optional[Dict[str, Any]]:
    """
    Extracts Theta (4-8Hz), Alpha (8-13Hz), and Beta (13-30Hz) from the buffer
    using Fast Fourier Transform (FFT) with Hanning window smoothing.
    """
    if not buffer or len(buffer) < sample_rate * 2:
        return None

    data = np.asarray(buffer, dtype=float)
    if data.ndim != 2:
        return None

    # Focus on the 4 primary Muse electrodes: TP9, AF7, AF8, TP10
    num_channels = min(4, data.shape[1])
    eeg = data[:, :num_channels]

    # Verify physical skin contact
    is_valid, quality = verify_electrode_contact(eeg)
    if not is_valid:
        return {
            "signal_quality": quality,
            "mind_state": "DISCONNECTED",
            "stress_level": 0.0,
            "focus_level": 0.0,
            "theta_power": 0.0,
            "alpha_power": 0.0,
            "beta_power": 0.0,
            "stress_ratio": 0.0,
            "engagement_ratio": 0.0,
            "confusion_ratio": 0.0,
            "fatigue_ratio": 0.0,
            "frontal_asymmetry": 0.0,
        }

    # Remove DC drift (zero-center) and apply Hanning window
    eeg_centered = eeg - np.mean(eeg, axis=0, keepdims=True)
    window = np.hanning(eeg_centered.shape[0])[:, None]
    eeg_windowed = eeg_centered * window

    # Compute FFT power spectral density
    freqs = np.fft.rfftfreq(eeg_windowed.shape[0], d=1.0 / sample_rate)
    fft_vals = np.fft.rfft(eeg_windowed, axis=0)
    power_per_channel = np.abs(fft_vals) ** 2  # Shape: (freq_bins, channels)

    def band_power(low: float, high: float, channels: Optional[List[int]] = None) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 1e-9
        if channels is not None:
            # Sum spectral power across frequency bins, then average across selected channels
            return float(np.mean(np.sum(power_per_channel[mask, :][:, channels], axis=0)))
        # Sum spectral power across frequency bins, then average across all 4 channels
        return float(np.mean(np.sum(power_per_channel[mask, :], axis=0)))

    # --- Extract the 3 Core Waves ---
    theta = band_power(4.0, 8.0)     # Theta: 4 - 8 Hz
    alpha = band_power(8.0, 13.0)    # Alpha: 8 - 13 Hz
    beta = band_power(13.0, 30.0)    # Beta:  13 - 30 Hz

    # Safeguard against zero division
    safe_alpha = max(alpha, 1e-6)
    safe_beta = max(beta, 1e-6)
    safe_theta = max(theta, 1e-6)

    # --- Physiological Ratios ---
    # 1. Stress Index: Beta / Alpha (High during cognitive tension / anxiety)
    stress_ratio = safe_beta / safe_alpha

    # 2. NASA Pope Engagement Index: Beta / (Theta + Alpha) (High during active focus)
    engagement_ratio = safe_beta / (safe_theta + safe_alpha)

    # 3. Confusion Index: Theta / Alpha (Elevates when working memory is overloaded)
    confusion_ratio = safe_theta / safe_alpha

    # 4. Mental Fatigue / Drowsiness Index: Theta / Beta (High when mentally drained)
    fatigue_ratio = safe_theta / safe_beta

    # 5. Frontal Alpha Asymmetry (AF7: ch 1, AF8: ch 2)
    # FAA = ln(Alpha_AF8) - ln(Alpha_AF7)
    # Negative FAA -> Relative right-frontal activation -> Stress / withdrawal
    # Positive FAA -> Relative left-frontal activation  -> Approach / positive mood
    if num_channels >= 3:
        alpha_af7 = band_power(8.0, 13.0, channels=[1])
        alpha_af8 = band_power(8.0, 13.0, channels=[2])
        faa = float(np.log(max(alpha_af8, 1e-6)) - np.log(max(alpha_af7, 1e-6)))
    else:
        faa = 0.0

    # --- Continuous 0 - 100% Normalized Scores ---
    # Baseline Beta/Alpha typically sits around 0.6 (relaxed) to 2.2 (high stress)
    norm_stress = ((stress_ratio - 0.5) / (2.2 - 0.5)) * 100.0
    stress_level = round(float(np.clip(norm_stress, 0.0, 100.0)), 1)

    # Baseline Pope Engagement typically sits around 0.4 (idle) to 1.4 (intense focus)
    norm_focus = ((engagement_ratio - 0.4) / (1.4 - 0.4)) * 100.0
    focus_level = round(float(np.clip(norm_focus, 0.0, 100.0)), 1)

    # --- Determine Categorical Cognitive Emotion ---
    # Physiological decision tree based on Theta (4-8Hz), Alpha (8-13Hz), and Beta (13-30Hz)

    # 1. Cognitive Confusion vs Fatigue: Theta surges and dominates the spectrum
    if theta > (1.2 * alpha) and theta > (1.2 * beta):
        # If accompanied by cognitive strain / beta tension -> Confused / working memory struggle
        if beta > alpha or stress_level >= 30.0:
            emotion = "CONFUSED"
        # Low arousal + dominant theta -> Mental fatigue / Drowsy
        else:
            emotion = "FATIGUED"

    # 2. Acute Stress & Anxiety: Beta surges and severely suppresses Alpha
    elif stress_level >= 70.0 or stress_ratio >= 1.75:
        emotion = "STRESSED"

    # 3. High Focus & Engagement: Active Beta with low Theta and moderate Alpha
    elif engagement_ratio >= 0.75 or (beta > (0.85 * alpha) and beta > (1.2 * theta)):
        emotion = "FOCUSED"

    # 4. Calm & Relaxed: Alpha clearly dominates over Beta and Theta
    elif alpha > (1.35 * beta) and alpha > theta and stress_level <= 35.0:
        emotion = "CALM"

    # 5. Baseline Resting State: Balanced rhythm
    else:
        emotion = "NEUTRAL"

    return {
        "signal_quality": "GOOD",
        "mind_state": emotion,
        "stress_level": stress_level,
        "focus_level": focus_level,
        "theta_power": round(theta, 3),
        "alpha_power": round(alpha, 3),
        "beta_power": round(beta, 3),
        "stress_ratio": round(stress_ratio, 3),
        "engagement_ratio": round(engagement_ratio, 3),
        "confusion_ratio": round(confusion_ratio, 3),
        "fatigue_ratio": round(fatigue_ratio, 3),
        "frontal_asymmetry": round(faa, 3),
    }


# ==============================================================================
# 3. DATABASE PERSISTENCE
# ==============================================================================

def save_to_db(metrics: Dict[str, Any], session_id: Optional[int] = None, student_id: Optional[str] = None):
    """
    Saves the computed state to the PostgreSQL database.
    - If `session_eeg_telemetry` exists, stores full numerical telemetry.
    - Also records to legacy `EEG_State` table for backwards compatibility.
    """
    state = metrics["mind_state"]
    if state == "DISCONNECTED":
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # 1. Try writing to legacy EEG_State table
        # Legacy table has check constraint: ('Neutral', 'CALM', 'STRESSED')
        # Map additional rich states to legacy categories if constraint is active
        legacy_state = state
        if legacy_state not in ("CALM", "STRESSED", "Neutral"):
            if legacy_state == "FOCUSED":
                legacy_state = "CALM"
            elif legacy_state in ("CONFUSED", "FATIGUED"):
                legacy_state = "Neutral"

        try:
            cur.execute("INSERT INTO EEG_State (mind_state) VALUES (%s);", (legacy_state,))
            conn.commit()
        except Exception:
            conn.rollback()

        # 2. If session telemetry table exists, write complete 3-wave numbers
        if session_id and student_id:
            try:
                cur.execute(
                    """
                    INSERT INTO session_eeg_telemetry 
                    (session_id, student_id, stress_score, alpha_power, beta_power, ratio, mind_state, signal_quality)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        session_id,
                        student_id,
                        metrics["stress_level"],
                        metrics["alpha_power"],
                        metrics["beta_power"],
                        metrics["stress_ratio"],
                        metrics["mind_state"],
                        metrics["signal_quality"],
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()

        cur.close()
        conn.close()
    except Exception as e:
        print(f"\n❌ DATABASE PERSISTENCE NOTICE: {e}\n")


# ==============================================================================
# 4. SIMULATION GENERATOR (FOR TESTING WITHOUT HEADBAND)
# ==============================================================================

def generate_synthetic_eeg_sample(mode: str, sample_index: int, srate: int = 256) -> List[float]:
    """Generates synthetic 4-channel Muse EEG data to simulate specific mental states."""
    t = sample_index / srate
    noise = np.random.normal(0, 3.0, 4)

    if mode == "CALM":
        # Dominant 10 Hz Alpha, low Beta and Theta
        alpha_wave = 28.0 * np.sin(2 * np.pi * 10.0 * t)
        beta_wave = 4.0 * np.sin(2 * np.pi * 20.0 * t)
        theta_wave = 6.0 * np.sin(2 * np.pi * 6.0 * t)
    elif mode == "STRESSED":
        # Spiking 24 Hz Beta, suppressed Alpha
        alpha_wave = 3.0 * np.sin(2 * np.pi * 10.0 * t)
        beta_wave = 35.0 * np.sin(2 * np.pi * 24.0 * t)
        theta_wave = 6.0 * np.sin(2 * np.pi * 6.0 * t)
    elif mode == "FOCUSED":
        # Strong active Beta, moderate Alpha, very low Theta (NASA Pope engagement)
        alpha_wave = 14.0 * np.sin(2 * np.pi * 10.0 * t)
        beta_wave = 18.0 * np.sin(2 * np.pi * 16.0 * t)
        theta_wave = 3.0 * np.sin(2 * np.pi * 6.0 * t)
    elif mode == "CONFUSED":
        # Surging Theta + active Beta tension
        alpha_wave = 6.0 * np.sin(2 * np.pi * 10.0 * t)
        beta_wave = 10.0 * np.sin(2 * np.pi * 18.0 * t)
        theta_wave = 35.0 * np.sin(2 * np.pi * 5.5 * t)
    else:  # NEUTRAL
        # Balanced Alpha, Beta, and Theta
        alpha_wave = 12.0 * np.sin(2 * np.pi * 10.0 * t)
        beta_wave = 11.0 * np.sin(2 * np.pi * 18.0 * t)
        theta_wave = 12.0 * np.sin(2 * np.pi * 6.0 * t)

    val = alpha_wave + beta_wave + theta_wave
    # Return 4 channels: TP9, AF7, AF8, TP10
    return [float(val + noise[0]), float(val + noise[1]), float(val + noise[2]), float(val + noise[3])]


# ==============================================================================
# 5. MAIN STREAMING LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Three-Wave (Theta, Alpha, Beta) EEG Streamer")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without hardware")
    parser.add_argument("--interval", type=int, default=10, help="DB save interval in seconds (default: 10s)")
    parser.add_argument("--session-id", type=int, default=None, help="Associated chat session ID")
    parser.add_argument("--student-id", type=str, default=None, help="Associated student ID")
    args = parser.parse_args()

    print("=" * 72)
    print("   🧠 EEG THREE-WAVE STREAMER (Theta | Alpha | Beta) Active   ")
    print("=" * 72)

    inlet = None
    channel_names = ["TP9", "AF7", "AF8", "TP10"]

    if not args.simulate and PYLSL_AVAILABLE:
        print("Looking for Muse EEG stream... (Ensure 'muselsl stream' is running)")
        streams = resolve_stream("type", "EEG")
        if streams:
            inlet = StreamInlet(streams[0])
            channel_names = get_channel_names(inlet)
            info = inlet.info()
            print(f"✅ Connected to Muse 2! Rate: {info.nominal_srate()} Hz | Nodes: {', '.join(channel_names)}")
        else:
            print("⚠️  No physical Muse LSL stream detected. Switching to SIMULATION mode.")
            args.simulate = True
    else:
        args.simulate = True
        print("ℹ️  Running in SIMULATION mode (generating realistic Theta/Alpha/Beta telemetry)...")

    sample_rate = 256
    window_samples = sample_rate * 4  # 4-second sliding analysis window
    buffer: List[List[float]] = []
    
    last_save_time = time.time()
    last_display_time = 0.0
    sample_index = 0
    sim_modes = ["CALM", "FOCUSED", "STRESSED", "CONFUSED", "NEUTRAL"]
    current_sim_idx = 0

    try:
        while True:
            # 1. Pull / Generate sample
            if args.simulate:
                mode = sim_modes[current_sim_idx]
                sample = generate_synthetic_eeg_sample(mode, sample_index, sample_rate)
                sample_index += 1
                time.sleep(1.0 / sample_rate)
                # Rotate simulation mode every 15 seconds to demonstrate all emotions
                if sample_index % (sample_rate * 15) == 0:
                    current_sim_idx = (current_sim_idx + 1) % len(sim_modes)
            else:
                sample, _ = inlet.pull_sample(timeout=1.0)
                if sample is None:
                    sys.stdout.write("\r⚠️  Waiting for incoming EEG packets from headband...")
                    sys.stdout.flush()
                    time.sleep(0.05)
                    continue

            buffer.append(sample)
            if len(buffer) > window_samples:
                buffer.pop(0)

            current_time = time.time()
            elapsed_save = current_time - last_save_time

            # 2. Live Console Display every 0.3 seconds
            if current_time - last_display_time >= 0.3:
                last_display_time = current_time
                metrics = extract_three_waves(buffer, sample_rate=sample_rate)

                if metrics and metrics["signal_quality"] == "GOOD":
                    state = metrics["mind_state"]
                    stress = metrics["stress_level"]
                    focus = metrics["focus_level"]
                    th = metrics["theta_power"]
                    al = metrics["alpha_power"]
                    be = metrics["beta_power"]
                    faa = metrics["frontal_asymmetry"]

                    # Emotion icon indicator
                    icon = {
                        "CALM": "🔵 CALM",
                        "FOCUSED": "🟢 FOCUSED",
                        "STRESSED": "🔴 STRESSED",
                        "CONFUSED": "🟡 CONFUSED",
                        "FATIGUED": "🟣 FATIGUED",
                        "NEUTRAL": "⚪ NEUTRAL",
                    }.get(state, state)

                    sys.stdout.write(
                        f"\r📡 [{elapsed_save:4.1f}s/{args.interval}s] "
                        f"State: {icon:<13} | "
                        f"Stress: {stress:5.1f}% | "
                        f"Focus: {focus:5.1f}% | "
                        f"Waves (θ:{th:4.1f} α:{al:4.1f} β:{be:4.1f}) | "
                        f"FAA:{faa:+0.2f}    "
                    )
                else:
                    sys.stdout.write("\r📡 Buffer initializing / verifying headband contact...       ")
                sys.stdout.flush()

            # 3. Save to DB on interval
            if elapsed_save >= args.interval:
                metrics = extract_three_waves(buffer, sample_rate=sample_rate)
                if metrics and metrics["signal_quality"] == "GOOD":
                    sys.stdout.write("\n")
                    print(f"💾 Saving to Database -> State: {metrics['mind_state']} | Stress Level: {metrics['stress_level']}% | Focus: {metrics['focus_level']}%")
                    save_to_db(metrics, session_id=args.session_id, student_id=args.student_id)
                last_save_time = current_time

    except KeyboardInterrupt:
        print("\n\n🛑 EEG Stream stopped by user.")


if __name__ == "__main__":
    main()
