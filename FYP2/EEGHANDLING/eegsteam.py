import time
import psycopg2
import numpy as np
from pylsl import StreamInlet, resolve_stream

# Your exact DB Configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "fyp_i",
    "user": "Sharjeel",
    "password": "Shamra349",
    "port": 5432
}


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



def analyze_30s_data(buffer):
    """Analyzes the collected 30 seconds of data to determine the state."""
    if not buffer:
        return "Neutral"

    data = np.array(buffer)

    # Simple averages (borrowed from your previous logic)
    # Assuming 4 channels, checking index 1 and 2
    alpha = np.mean(data[:, 1])
    beta = np.mean(data[:, 2])

    # Must exactly match your SQL CHECK constraint ('Neutral', 'CALM', 'STRESSED')
    if beta > 0.7:
        return "STRESSED"
    elif alpha > 0.6:
        return "CALM"
    else:
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

        # 2. Print data to the console continuously
        print(f"Streaming Raw Data: {sample}")

        # 3. Add to our buffer for analysis
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