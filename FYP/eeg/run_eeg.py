from FYP.eeg.stream import connect_muse, get_sample
from FYP.eeg.buffer import EEGBuffer
from FYP.eeg.processor import extract_features
from FYP.eeg.analyzer import analyze_state
from FYP.eeg.sender import send_eeg

def run():
    inlet = connect_muse()
    buffer = EEGBuffer(window_seconds=10)

    while True:
        sample = get_sample(inlet)
        buffer.add_sample(sample)

        if buffer.is_ready():
            data = buffer.get_and_reset()

            features = extract_features(data)
            state = analyze_state(features)

            print("EEG State:", state)

            send_eeg(features, state)


if __name__ == "__main__":
    run()