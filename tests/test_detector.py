import unittest

from account_ar.config import Settings
from account_ar.detector import associate_accounts
from account_ar.ordering import rank_accounts
from tests.helpers import det


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def test_same_line_full_phrase(self):
        # Hebrew RTL: "מספר חשבון" (two OCR boxes) on the right, number to its LEFT.
        dets = [
            det("מספר", 330, 100, 420, 130),
            det("חשבון", 210, 100, 320, 130),
            det("292039", 80, 100, 200, 130),
        ]
        accounts = associate_accounts(dets, self.s)
        self.assertEqual(sorted(a.digits for a in accounts), ["292039"])

    def test_account_word_alone_is_enough(self):
        # Robust to OCR dropping the short "מספר": a line with חשבון + a number is
        # detected (חשבון is the word that marks an account line).
        dets = [
            det("חשבון", 210, 100, 320, 130),
            det("292039", 80, 100, 200, 130),
        ]
        accounts = associate_accounts(dets, self.s)
        self.assertEqual([a.digits for a in accounts], ["292039"])

    def test_savings_line_ignored(self):
        # "מספר חסכון" (savings) must NOT be detected — חסכון is not חשבון.
        dets = [
            det("מספר", 330, 100, 420, 130),
            det("חסכון", 210, 100, 320, 130),
            det("445566", 80, 100, 200, 130),
        ]
        self.assertEqual(associate_accounts(dets, self.s), [])

    def test_require_number_word_mode(self):
        # Opt-in strict mode: bare "חשבון" no longer enough, "מספר"/"מס" required.
        self.s.require_number_word = True
        bare = [det("חשבון", 210, 100, 320, 130), det("292039", 80, 100, 200, 130)]
        self.assertEqual(associate_accounts(bare, self.s), [])
        full = [det("מספר", 330, 100, 420, 130), det("חשבון", 210, 100, 320, 130),
                det("292039", 80, 100, 200, 130)]
        self.assertEqual([a.digits for a in associate_accounts(full, self.s)], ["292039"])

    def test_wrong_side_number_ignored(self):
        # A number on the WRONG side of the label (to its right, in RTL) is ignored.
        dets = [
            det("מספר", 330, 100, 420, 130),
            det("חשבון", 210, 100, 320, 130),
            det("12345", 440, 100, 540, 130),
        ]
        self.assertEqual(associate_accounts(dets, self.s), [])

    def test_main_account_label_same_line(self):
        # "מס חשבון ראשי" is recognised; the number after it on the line is taken,
        # and "ראשי" (other content) is ignored.
        dets = [
            det("מס חשבון ראשי", 210, 100, 380, 130),
            det("299868", 80, 100, 200, 130),
        ]
        accounts = associate_accounts(dets, self.s)
        self.assertEqual([a.digits for a in accounts], ["299868"])

    def test_branch_line_ignored(self):
        # "מספר סניף 16" has a number word but no חשבון -> not an account line.
        dets = [
            det("מספר", 330, 100, 420, 130),
            det("סניף", 210, 100, 320, 130),
            det("16", 150, 100, 200, 130),
        ]
        self.assertEqual(associate_accounts(dets, self.s), [])

    def test_below_is_not_same_line(self):
        # Only numbers on the SAME line as the label count; one below is ignored.
        dets = [
            det("מספר חשבון", 100, 100, 300, 130),
            det("290134", 150, 150, 260, 180),
        ]
        self.assertEqual(associate_accounts(dets, self.s), [])

    def test_combined_box(self):
        # One OCR box holds the whole phrase and the number.
        dets = [det("מספר חשבון 292039", 100, 100, 380, 130)]
        accounts = associate_accounts(dets, self.s)
        self.assertEqual([a.digits for a in accounts], ["292039"])

    def test_unlabeled_rejected_by_default(self):
        dets = [det("250620", 100, 100, 200, 130)]  # looks like a date, no label
        accounts = associate_accounts(dets, self.s)
        self.assertEqual(accounts, [])

    def test_unlabeled_accepted_when_enabled(self):
        self.s.accept_unlabeled = True
        dets = [det("250620", 100, 100, 200, 130)]
        accounts = associate_accounts(dets, self.s)
        self.assertEqual([a.digits for a in accounts], ["250620"])

    def test_full_pipeline_three_accounts(self):
        # Three labelled accounts on three rows -> correct order after ranking.
        # RTL: "מספר חשבון" label on the right, number to its left on each row.
        dets = [
            det("מספר", 330, 100, 420, 130), det("חשבון", 210, 100, 320, 130), det("291039", 80, 100, 200, 130),
            det("מספר", 330, 160, 420, 190), det("חשבון", 210, 160, 320, 190), det("292039", 80, 160, 200, 190),
            det("מספר", 330, 220, 420, 250), det("חשבון", 210, 220, 320, 250), det("290134", 80, 220, 200, 250),
        ]
        accounts = rank_accounts(associate_accounts(dets, self.s), self.s)
        self.assertEqual([a.digits for a in accounts], ["290134", "291039", "292039"])
        self.assertEqual([a.rank for a in accounts], [1, 2, 3])

    def test_dedupe_same_number(self):
        dets = [
            det("מספר", 330, 100, 420, 130),
            det("חשבון", 210, 100, 320, 130),
            det("292039", 80, 100, 200, 130, conf=0.7),
            det("292039", 80, 100, 200, 130, conf=0.95),
        ]
        accounts = associate_accounts(dets, self.s)
        self.assertEqual(len(accounts), 1)
        self.assertAlmostEqual(accounts[0].confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
