import sys
import time
import psycopg2
import numpy as np
from pylsl import StreamInlet, resolve_stream
from FYP2.core.config import DB_CONFIG
#defaultnodes names
DEFAULT_MUSE_5NODE_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]
#second step to get all nodes name if available
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


def save_to_db(mind_state):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO EEG_State (mind_State) VALUES (%s);"
        cursor.execute(query, (mind_state,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"\n=========================================")
        print(f"✅ 30 SECONDS UP! SAVED TO DB: {mind_state}")
        print(f"=========================================\n")
    except Exception as e:
        print(f"\n❌ DATABASE ERROR: {e}\n")


def analyze_30s_data(buffer, sample_rate=256):
    if not buffer:
        return "Neutral"
    data = np.asarray(buffer, dtype=float)
    if data.ndim != 2 or data.shape[0] < sample_rate * 5:
        return "Neutral"
    num_channels = min(5, data.shape[1])
    eeg = data[:, :num_channels]
    eeg -= np.mean(eeg, axis=0, keepdims=True)
    eeg *= np.hanning(eeg.shape[0])[:, None]

    frequencies = np.fft.rfftfreq(eeg.shape[0], d=1 / sample_rate)
    power = np.mean(np.abs(np.fft.rfft(eeg, axis=0)) ** 2, axis=1)

    def band_power(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        return float(np.sum(power[mask]))

    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    ratio = beta / max(alpha, 1e-9)

    if ratio > 1.6:
        return "STRESSED"
    elif ratio < 0.8:
        return "CALM"
    return "Neutral"


def main():
    print("Looking for Muse EEG stream... (Make sure 'muselsl stream' is running)")
    streams = resolve_stream('type', 'EEG')

    if not streams:
        raise Exception("No EEG stream found.")

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    channel_names = get_channel_names(inlet)

    print("Connected to Muse 2! Starting continuous stream...")
    print(f"Stream Name: {info.name()} | Rate: {info.nominal_srate()} Hz | Nodes ({len(channel_names)}): {', '.join(channel_names)}\n")

    buffer = []
    last_save_time = time.time()
    last_display_time = 0

    while True:
        # 1. Pull live sample with timeout to prevent blocking indefinitely
        sample, timestamp = inlet.pull_sample(timeout=1.0)

        if sample is None:
            sys.stdout.write("\r⚠️  Waiting for EEG data stream from Muse headband...")
            sys.stdout.flush()
            time.sleep(0.1)
            continue

        buffer.append(sample)
        current_time = time.time()
        elapsed = current_time - last_save_time

        # 2. Display real-time incoming signal values every 0.25 seconds
        if current_time - last_display_time >= 0.25:
            last_display_time = current_time

            # Format 5-node channel signals (in µV)
            signals_str = " | ".join(
                f"{name}: {val:+.1f}µV" for name, val in zip(channel_names, sample)
            )

            # Compute live state hint from recent 5 seconds of stream buffer if available
            live_state = "Initializing..."
            if len(buffer) >= 256 * 5:
                live_state = analyze_30s_data(buffer[-1280:])

            # Overwrite line live in console
            sys.stdout.write(
                f"\r📡 [{elapsed:4.1f}s / 30.0s] Signals: {signals_str} | Live: {live_state}    "
            )
            sys.stdout.flush()

        # 3. Save to DB every 30 seconds
        if elapsed >= 30:
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Figure out state from the 30-second buffer
            current_state = analyze_30s_data(buffer)

            # Save state to PostgreSQL DB
            save_to_db(current_state)

            # Reset timer and clear buffer for next cycle
            last_save_time = time.time()
            buffer = []


if __name__ == "__main__":
    main()

