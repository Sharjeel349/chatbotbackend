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
        # 1. Neutral (0.75 <= Ratio < 1.15)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=9.0, theta=10.0)
        self.assertEqual(code, STATE_NEUTRAL)
        self.assertEqual(name, "Neutral")
        self.assertAlmostEqual(bar, 0.90)

        # 2. Calm (Ratio < 0.75)
        code, name, bar, ear = classify_brain_state(alpha=20.0, beta=6.0, theta=8.0)
        self.assertEqual(code, STATE_CALM)
        self.assertEqual(name, "Calm")
        self.assertAlmostEqual(bar, 0.30)

        # 3. Focused (1.15 <= Ratio < 1.65)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=13.5, theta=6.0)
        self.assertEqual(code, STATE_FOCUSED)
        self.assertEqual(name, "Focused")
        self.assertAlmostEqual(bar, 1.35)

        # 4. Mild Stress (1.65 <= Ratio < 2.40)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=19.0, theta=10.0)
        self.assertEqual(code, STATE_MILD_STRESS)
        self.assertEqual(name, "Mild Stress")
        self.assertAlmostEqual(bar, 1.90)

        # 5. Moderate Stress (2.40 <= Ratio < 3.00)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=26.0, theta=10.0)
        self.assertEqual(code, STATE_MODERATE_STRESS)
        self.assertEqual(name, "Moderate Stress")
        self.assertAlmostEqual(bar, 2.60)

        # 6. Severe Stress (Ratio >= 3.00)
        code, name, bar, ear = classify_brain_state(alpha=10.0, beta=35.0, theta=8.0)
        self.assertEqual(code, STATE_SEVERE_STRESS)
        self.assertEqual(name, "Severe Stress")
        self.assertAlmostEqual(bar, 3.50)

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

