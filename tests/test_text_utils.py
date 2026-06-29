import unittest

from account_ar.config import Settings
from account_ar.text_utils import (
    extract_number_candidates,
    is_label,
    line_label_match,
    normalize_hebrew,
    strip_to_digits,
)


class TestTextUtils(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def test_normalize_strips_geresh_and_punct(self):
        # geresh + dot removed, final nun folded to base form: "מס׳ ח.ן" -> "מס חנ"
        self.assertEqual(normalize_hebrew("מס׳ ח.ן"), "מס חנ")

    def test_normalize_reversed_account_word(self):
        # Tesseract often emits Hebrew visually-reversed; reversing recovers it.
        # "|ובשח" (OCR of חשבון) -> strip noise -> "ובשח"; reversed -> "חשבו".
        self.assertEqual(normalize_hebrew("|ובשח"), "ובשח")

    def test_strip_to_digits(self):
        self.assertEqual(strip_to_digits("12-345/678"), "12345678")

    def test_extract_plain_number(self):
        self.assertEqual(extract_number_candidates("292039", self.s), ["292039"])

    def test_extract_with_separators(self):
        self.assertEqual(extract_number_candidates("12-345678", self.s), ["12345678"])

    def test_extract_from_mixed_text(self):
        self.assertEqual(extract_number_candidates("חשבון 292039", self.s), ["292039"])

    def test_too_short_rejected(self):
        self.assertEqual(extract_number_candidates("12", self.s), [])

    def test_is_label_for_keywords(self):
        for kw in ["חשבון", "מס׳ חשבון", "חשבון בנק", "מס ח.ן", "סניף/חשבון"]:
            self.assertTrue(is_label(kw, self.s), f"expected label: {kw}")

    def test_is_label_reversed_and_noisy(self):
        # Exactly how Tesseract returned them (visual order + OCR noise).
        for ocr in ["|ובשח", "ןובשח", "|ובשח/ףינס"]:
            self.assertTrue(is_label(ocr, self.s), f"expected label: {ocr}")

    def test_non_label_rejected(self):
        self.assertFalse(is_label("שלום", self.s))
        self.assertFalse(is_label("292039", self.s))

    def test_only_account_fields_not_branch_savings(self):
        # The user wants ONLY account-number fields — branch / savings / date / amount
        # numbers must not be tagged.
        for non_account in ["מספר סניף", "סניף", "חסכון", "מספר חסכון", "תאריך", "סכום"]:
            self.assertFalse(is_label(non_account, self.s),
                             f"should NOT be an account label: {non_account}")

    # --- line-level matching (the actual detection gate) ---

    def test_line_label_match_keys_on_account_word(self):
        # The account noun חשבון marks an account line, with or without "מספר".
        self.assertIsNotNone(line_label_match(["מספר", "חשבון", "299868"], self.s))
        self.assertIsNotNone(line_label_match(["מס", "חשבון", "ראשי"], self.s))
        self.assertIsNotNone(line_label_match(["חשבון"], self.s))  # robust: bare חשבון

    def test_line_label_match_rejects_other_fields(self):
        # Branch and savings have a different distinctive word, so they never match.
        self.assertIsNone(line_label_match(["מספר", "סניף", "16"], self.s))
        self.assertIsNone(line_label_match(["מספר", "חסכון"], self.s))
        self.assertIsNone(line_label_match(["תאריך", "21/06/2023"], self.s))

    def test_line_label_match_reversed_ocr(self):
        # Tesseract emitting Hebrew visually reversed still matches.
        self.assertIsNotNone(line_label_match(["רפסמ", "ןובשח"], self.s))


if __name__ == "__main__":
    unittest.main()
