# 1. module imports
import unittest
from ocr_benchmark.metrics import (
    compute_cer,
    compute_wer,
    compute_ned,
    compute_character_f1,
    compute_word_f1,
    compute_diacritic_error_rate,
    compute_bracket_f1,
    compute_sentence_matching,
    compute_word_per_sentence,
    evaluate_transcription,
)

# 2. constant definitions
SAMPLE_GT = "1. i-na a-lim(KI) KÀ-ŠU-UM\n2. DUMU ša-lim-a-ḫum"
SAMPLE_PRED_EXACT = "1. i-na a-lim(KI) KÀ-ŠU-UM\n2. DUMU ša-lim-a-ḫum"
SAMPLE_PRED_UNORDERED = "2. DUMU ša-lim-a-ḫum\n1. i-na a-lim(KI) KÀ-ŠU-UM"
SAMPLE_PRED_DIACRITIC_ERROR = "1. i-na a-lim(KI) KA-SU-UM\n2. DUMU sa-lim-a-hum"

# 3. class definitions
class TestOCRMetrics(unittest.TestCase):

    def test_cer_exact(self):
        cer = compute_cer(SAMPLE_GT, SAMPLE_PRED_EXACT)
        self.assertEqual(cer, 0.0)

    def test_cer_case_insensitive(self):
        cer_sens = compute_cer("KÀ-ŠU-UM", "kà-šu-um", case_sensitive=True)
        cer_insens = compute_cer("KÀ-ŠU-UM", "kà-šu-um", case_sensitive=False)
        self.assertGreater(cer_sens, 0.0)
        self.assertEqual(cer_insens, 0.0)

    def test_diacritic_error_rate(self):
        der_exact = compute_diacritic_error_rate(SAMPLE_GT, SAMPLE_PRED_EXACT)
        der_error = compute_diacritic_error_rate(SAMPLE_GT, SAMPLE_PRED_DIACRITIC_ERROR)
        self.assertEqual(der_exact, 0.0)
        self.assertEqual(der_error, 1.0)

    def test_bracket_f1(self):
        bracket_f1_exact = compute_bracket_f1(SAMPLE_GT, SAMPLE_PRED_EXACT)
        bracket_f1_missing = compute_bracket_f1(SAMPLE_GT, "1. i-na a-lim KI KÀ-ŠU-UM")
        self.assertEqual(bracket_f1_exact, 1.0)
        self.assertEqual(bracket_f1_missing, 0.0)

    def test_sentence_matching_unordered(self):
        result = compute_sentence_matching(
            [l.strip() for l in SAMPLE_GT.splitlines()],
            [l.strip() for l in SAMPLE_PRED_UNORDERED.splitlines()],
        )
        self.assertEqual(result.ordered_exact_ratio, 0.0)
        self.assertEqual(result.unordered_exact_ratio, 1.0)
        self.assertGreater(result.unordered_hungarian_ratio, 0.99)

    def test_full_evaluation(self):
        metrics = evaluate_transcription(SAMPLE_GT, SAMPLE_PRED_EXACT)
        self.assertEqual(metrics.cer, 0.0)
        self.assertEqual(metrics.wer, 0.0)
        self.assertEqual(metrics.domain_metrics.diacritic_error_rate, 0.0)
        self.assertEqual(metrics.domain_metrics.bracket_f1, 1.0)

# 4. function definitions
# Unittest test runner

# 5. class instantiation
if __name__ == "__main__":
    unittest.main()
