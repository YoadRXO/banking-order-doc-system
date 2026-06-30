"""Temporal accumulation: one readable frame should keep an account on screen
across the many unreadable frames that follow (glare/blur/motion), then expire."""
import unittest

from account_ar.pipeline import AccountTracker
from account_ar.types import AccountNumber, Detection


def _acct(digits, conf=0.9):
    poly = [(0, 0), (10, 0), (10, 5), (0, 5)]
    return AccountNumber(digits=digits, detection=Detection(digits, conf, poly), confidence=conf)


class TestAccountTracker(unittest.TestCase):
    def test_sticks_then_expires(self):
        tr = AccountTracker(ttl=2.5)
        self.assertEqual([a.digits for a in tr.update([_acct("299868")], now=0.0)], ["299868"])
        # No detections on the next (bad) frames — the account is remembered.
        self.assertEqual([a.digits for a in tr.update([], now=1.0)], ["299868"])
        self.assertEqual([a.digits for a in tr.update([], now=2.0)], ["299868"])
        # Past the ttl → forgotten.
        self.assertEqual(tr.update([], now=3.0), [])

    def test_resighting_refreshes_ttl_and_box(self):
        tr = AccountTracker(ttl=2.5)
        tr.update([_acct("299868", conf=0.5)], now=0.0)
        fresh = _acct("299868", conf=0.9)
        out = tr.update([fresh], now=2.0)  # re-seen before expiry
        self.assertEqual([a.digits for a in out], ["299868"])
        self.assertIs(out[0].detection, fresh.detection)  # newest (freshest) box wins
        self.assertEqual([a.digits for a in tr.update([], now=4.0)], ["299868"])  # ttl refreshed

    def test_multiple_accounts_tracked_independently(self):
        tr = AccountTracker(ttl=2.0)
        tr.update([_acct("290134")], now=0.0)
        out = tr.update([_acct("292039")], now=1.0)
        self.assertEqual(sorted(a.digits for a in out), ["290134", "292039"])
        out = tr.update([], now=2.5)  # 290134 expired (seen at 0.0), 292039 still alive
        self.assertEqual([a.digits for a in out], ["292039"])

    def test_clear(self):
        tr = AccountTracker(ttl=10.0)
        tr.update([_acct("299868")], now=0.0)
        tr.clear()
        self.assertEqual(tr.update([], now=0.1), [])

    def test_confirm_sightings_hides_one_off_misread(self):
        # SAFETY: with confirm_sightings=2 a number must be read twice before it shows,
        # so a single rotated/glare misread never reaches the screen.
        tr = AccountTracker(ttl=3.0, confirm_sightings=2)
        self.assertEqual(tr.update([_acct("898662")], now=0.0), [])      # first sighting: hidden
        out = tr.update([_acct("898662")], now=1.0)                       # second: confirmed
        self.assertEqual([a.digits for a in out], ["898662"])

    def test_confirm_sightings_one_off_then_gone(self):
        tr = AccountTracker(ttl=2.0, confirm_sightings=2)
        self.assertEqual(tr.update([_acct("898662")], now=0.0), [])      # seen once, hidden
        self.assertEqual(tr.update([], now=1.0), [])                      # not re-seen, still hidden
        self.assertEqual(tr.update([], now=3.0), [])                      # expired, never shown


if __name__ == "__main__":
    unittest.main()
