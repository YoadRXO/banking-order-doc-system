import unittest

from account_ar.config import Settings
from account_ar.ordering import rank_accounts
from account_ar.types import AccountNumber, Detection


def _acc(digits: str) -> AccountNumber:
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return AccountNumber(digits=digits, detection=Detection(digits, 0.9, poly))


class TestOrdering(unittest.TestCase):
    def test_worked_example_ascending(self):
        # The exact example from the spec.
        accounts = [_acc("291039"), _acc("292039"), _acc("290134")]
        ranked = rank_accounts(accounts, Settings())
        self.assertEqual([a.digits for a in ranked], ["290134", "291039", "292039"])
        self.assertEqual([a.rank for a in ranked], [1, 2, 3])

    def test_descending(self):
        s = Settings()
        s.ascending = False
        accounts = [_acc("291039"), _acc("292039"), _acc("290134")]
        ranked = rank_accounts(accounts, s)
        self.assertEqual([a.digits for a in ranked], ["292039", "291039", "290134"])


if __name__ == "__main__":
    unittest.main()
