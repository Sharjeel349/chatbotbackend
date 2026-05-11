import time

class EEGBuffer:
    def __init__(self, window_seconds=10):
        self.window_seconds = window_seconds
        self.buffer = []
        self.start_time = time.time()

    def add_sample(self, sample):
        self.buffer.append(sample)

    def is_ready(self):
        return (time.time() - self.start_time) >= self.window_seconds

    def get_and_reset(self):
        data = self.buffer
        self.buffer = []
        self.start_time = time.time()
        return data