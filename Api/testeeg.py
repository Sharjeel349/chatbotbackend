import os

# 🔥 IMPORTANT: Force LSL path (Mac fix)
os.environ['PYLSL_LIB'] = '/usr/local/opt/lsl/lib/liblsl.dylib'

from pylsl import StreamInlet, resolve_stream

print("🔍 Looking for EEG stream...")

# Find EEG stream
streams = resolve_stream('type', 'EEG')

if not streams:
    print("❌ No EEG stream found.")
    print("👉 Make sure you ran: python -m muselsl stream")
    exit()

print("✅ Stream found!")

# Connect to stream
inlet = StreamInlet(streams[0])

print("📡 Receiving EEG data...\n")

# Continuously print data
while True:
    sample, timestamp = inlet.pull_sample()

    print(f"Time: {timestamp}")
    print(f"EEG Channels: {sample}")
    print("-" * 50)