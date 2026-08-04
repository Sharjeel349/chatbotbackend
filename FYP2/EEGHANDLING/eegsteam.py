import time
import psycopg2
import numpy as np
from pylsl import StreamInlet, resolve_stream
from FYP2.core.config import DB_CONFIG


def save_to_db(mind_state):
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Insert query (timestamp and ID are handled automatically by Postgres)
        query = "INSERT INTO EEG_State (mind_State) VALUES (%s);"
        cursor.execute(query, (mind_state,))

        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()

        print(f"\n=========================================")
        print(f"✅ 30 SECONDS UP! SAVED TO DB: {mind_state}")
        print(f"=========================================\n")

    except Exception as e:
        print(f"\n❌ DATABASE ERROR: {e}\n")



def analyze_30s_data(buffer, sample_rate=256):
    """Estimate a coarse state from relative alpha and beta band power.

    This is a research heuristic, not a medical or emotional diagnosis.
    Production use still needs artifact rejection and per-user calibration.
    """
    if not buffer:
        return "Neutral"

    data = np.asarray(buffer, dtype=float)
    if data.ndim != 2 or data.shape[0] < sample_rate * 5:
        return "Neutral"

    eeg = data[:, : min(4, data.shape[1])]
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

    #streams= pylsl.resolve_stream("type", "EEG")

    if not streams:
        raise Exception("No EEG stream found.")

    inlet = StreamInlet(streams[0])
    print("Connected to Muse! Starting continuous stream...\n")

    buffer = []
    last_save_time = time.time()

    while True:
        # 1. Pull the live sample
        sample, timestamp = inlet.pull_sample()

        # Avoid logging every raw sample; it is slow and exposes sensitive data.
        buffer.append(sample)

        current_time = time.time()

        # 4. Check if 30 seconds have passed
        if current_time - last_save_time >= 30:
            # Figure out the state from the last 30 seconds of data
            current_state = analyze_30s_data(buffer)

            # Save it to the database
            save_to_db(current_state)

            # Reset the timer and clear the buffer for the next 30 seconds
            last_save_time = current_time
            buffer = []


if __name__ == "__main__":
    main()
