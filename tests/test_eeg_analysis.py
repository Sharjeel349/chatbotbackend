import unittest

import numpy as np

from FYP2.EEGHANDLING.eegsteam import analyze_30s_data, get_channel_names, DEFAULT_MUSE_5NODE_CHANNELS


class EEGAnalysisTests(unittest.TestCase):
    def _signal(self, frequency, num_channels=4):
        sample_rate = 256
        seconds = 5
        time = np.arange(sample_rate * seconds) / sample_rate
        channel = np.sin(2 * np.pi * frequency * time)
        return np.column_stack([channel] * num_channels).tolist()

    def test_alpha_dominant_signal_is_calm(self):
        self.assertEqual(analyze_30s_data(self._signal(10)), "CALM")

    def test_beta_dominant_signal_is_stressed(self):
        self.assertEqual(analyze_30s_data(self._signal(20)), "STRESSED")

    def test_5channel_muse2_signal_analysis(self):
        # Test with 5 nodes headband (TP9, AF7, AF8, TP10, AUX)
        self.assertEqual(analyze_30s_data(self._signal(10, num_channels=5)), "CALM")
        self.assertEqual(analyze_30s_data(self._signal(20, num_channels=5)), "STRESSED")

    def test_default_channel_names(self):
        class MockInlet:
            class MockInfo:
                def channel_count(self):
                    return 5
                def desc(self):
                    class EmptyDesc:
                        def child(self, name):
                            return self
                        def empty(self):
                            return True
                    return EmptyDesc()
            def info(self):
                return self.MockInfo()

        names = get_channel_names(MockInlet())
        self.assertEqual(names, ["TP9", "AF7", "AF8", "TP10", "AUX"])


if __name__ == "__main__":
    unittest.main()

