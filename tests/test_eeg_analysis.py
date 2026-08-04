import unittest

import numpy as np

from FYP2.EEGHANDLING.eegsteam import analyze_30s_data


class EEGAnalysisTests(unittest.TestCase):
    def _signal(self, frequency):
        sample_rate = 256
        seconds = 5
        time = np.arange(sample_rate * seconds) / sample_rate
        channel = np.sin(2 * np.pi * frequency * time)
        return np.column_stack([channel] * 4).tolist()

    def test_alpha_dominant_signal_is_calm(self):
        self.assertEqual(analyze_30s_data(self._signal(10)), "CALM")

    def test_beta_dominant_signal_is_stressed(self):
        self.assertEqual(analyze_30s_data(self._signal(20)), "STRESSED")


if __name__ == "__main__":
    unittest.main()
