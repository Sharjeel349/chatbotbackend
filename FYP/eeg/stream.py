
import os

import os
os.environ['PYLSL_LIB'] = '/usr/local/opt/lsl/lib/liblsl.dylib'
from pylsl import StreamInlet, resolve_stream

def connect_muse():
    print("Looking for Muse EEG stream...")
    streams = resolve_stream('type', 'EEG')

    if not streams:
        raise Exception("No EEG stream found. Run: muselsl stream")

    inlet = StreamInlet(streams[0])
    print("Connected to Muse")
    return inlet


def get_sample(inlet):
    sample, timestamp = inlet.pull_sample()
    return sample




# from pylsl import resolve_stream
# from pythonosc import dispatcher
# from pythonosc import osc_server
#
# # This function triggers every time EEG data arrives
# def eeg_handler(address, *args):
#     # args contains the 4 or 5 sensor values
#     print(f"Received EEG data on {address}: {args}")
#
# def connect_muse_osc():
#     # Create a dispatcher to handle incoming data
#     disp = dispatcher.Dispatcher()
#     # "/muse/eeg" is the default address Mind Monitor sends data to
#     disp.map("/muse/eeg", eeg_handler)
#
#     # Listen on your Mac's IP (or 0.0.0.0 for all) on the default port 5000
#     server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 5000), disp)
#     print("Listening for Mind Monitor OSC stream on port 5000...")
#     server.serve_forever()

