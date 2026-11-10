import unittest
import numpy as np

from FYP2.EEGHANDLING.updated_eeg_schema import (
    STATE_NEUTRAL,
    STATE_CALM,
    STATE_FOCUSED,
    STATE_MILD_STRESS,
    STATE_MODERATE_STRESS,
    STATE_SEVERE_STRESS,
    STATE_NAMES,
    classify_brain_state,
    analyze_eeg_buffer,
)


class TestUpdatedEEGSchema(unittest.TestCase):
    def test_state_mapping_constants(self):
        self.assertEqual(STATE_NAMES[STATE_NEUTRAL], "Neutral")
        self.assertEqual(STATE_NAMES[STATE_CALM], "Calm")
        self.assertEqual(STATE_NAMES[STATE_FOCUSED], "Focused")
        self.assertEqual(STATE_NAMES[STATE_MILD_STRESS], "Mild Stress")
        self.assertEqual(STATE_NAMES[STATE_MODERATE_STRESS], "Moderate Stress")
        self.assertEqual(STATE_NAMES[STATE_SEVERE_STRESS], "Severe Stress")

    def test_state_classification_cases(self):
        # 1. Neutral (balanced resting baseline)
        code, name, bar, ear = classify_brain_state(alpha=12.0, beta=10.0, theta=14.0)
        self.assertEqual(code, STATE_NEUTRAL)
        self.assertEqual(name, "Neutral")

        # 2. Calm (dominant alpha)
        code, name, bar, ear = classify_brain_state(alpha=25.0, beta=7.0, theta=10.0)
        self.assertEqual(code, STATE_CALM)
        self.assertEqual(name, "Calm")

        # 3. Focused (high cognitive engagement without stress)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=14.0, theta=8.0)
        self.assertEqual(code, STATE_FOCUSED)
        self.assertEqual(name, "Focused")

        # 4. Mild Stress (1.5 <= BAR < 2.0)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=16.0, theta=10.0)
        self.assertEqual(code, STATE_MILD_STRESS)
        self.assertEqual(name, "Mild Stress")

        # 5. Moderate Stress (2.0 <= BAR < 2.8)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=22.0, theta=10.0)
        self.assertEqual(code, STATE_MODERATE_STRESS)
        self.assertEqual(name, "Moderate Stress")

        # 6. Severe Stress (BAR >= 2.8)
        code, name, bar, ear = classify_brain_state(alpha=8.0, beta=26.0, theta=8.0)
        self.assertEqual(code, STATE_SEVERE_STRESS)
        self.assertEqual(name, "Severe Stress")

    def test_analyze_eeg_buffer_output_structure(self):
        # 128 samples across 4 channels
        synthetic_buffer = np.random.randn(128, 4).tolist()
        result = analyze_eeg_buffer(synthetic_buffer, sample_rate=256)

        self.assertIn(result.state_code, [1, 2, 3, 4, 5, 6])
        self.assertIn(result.state_name, STATE_NAMES.values())
        self.assertGreaterEqual(result.theta, 0.0)
        self.assertGreaterEqual(result.alpha, 0.0)
        self.assertGreaterEqual(result.beta, 0.0)

        d = result.to_dict()
        self.assertEqual(d["state_code"], result.state_code)
        self.assertEqual(d["state_name"], result.state_name)

    def test_compute_sliding_live_metrics(self):
        from FYP2.EEGHANDLING.updated_eeg_schema import compute_sliding_live_metrics

        synthetic = np.random.randn(160, 4).tolist()
        live_rows, remaining_tail = compute_sliding_live_metrics(synthetic, window_size=128, step_size=32)

        self.assertGreater(len(live_rows), 0)
        self.assertIn("mind_state", live_rows[0])
        self.assertIn(live_rows[0]["mind_state"], STATE_NAMES.values())
        self.assertIn("ratio", live_rows[0])
        self.assertEqual(len(remaining_tail), 96)

    def test_process_and_save_eeg_stream_empty(self):
        from FYP2.EEGHANDLING.updated_eeg_schema import process_and_save_eeg_stream

        resp = process_and_save_eeg_stream([])
        self.assertEqual(resp["status"], "empty")


if __name__ == "__main__":
    unittest.main()

