"""Tests for multi-document grouping + stacking order (Phase 4.5).

Which paper goes on TOP is decided by the account NUMBER, not its position on the
table. The headline case: right paper 29200, left paper 29201 -> put 29200 on top.
"""
import unittest

from account_ar.config import Settings
from account_ar.grouping import group_documents, stack_instruction
from account_ar.types import AccountNumber
from tests.helpers import det


def acc(digits: str, x1, y1, x2, y2, label=None, conf: float = 0.9) -> AccountNumber:
    return AccountNumber(digits=digits, detection=det(digits, x1, y1, x2, y2, conf),
                         label_text=label, confidence=conf)


class TestGrouping(unittest.TestCase):
    def setUp(self):
        self.s = Settings()
        self.s.stack_order_enabled = True

    def test_two_papers_right_smaller_goes_on_top(self):
        # The request's example: right shows 29200, left shows 29201. Held apart, so a
        # clear horizontal gap between the two number boxes.
        right = acc("29200", 800, 100, 920, 130)
        left = acc("29201", 100, 100, 220, 130)
        docs = group_documents([right, left], self.s)

        self.assertEqual(len(docs), 2)                       # two documents
        by_pos = {d.stack_position: d.digits for d in docs}
        self.assertEqual(by_pos[1], "29200")                 # top of the stack
        self.assertEqual(by_pos[2], "29201")
        self.assertEqual(stack_instruction(docs),
                         "Put 29200 on top, then 29201 under it.")

    def test_descending_flips_top(self):
        self.s.ascending = False
        docs = group_documents(
            [acc("29200", 800, 100, 920, 130), acc("29201", 100, 100, 220, 130)], self.s)
        top = next(d for d in docs if d.stack_position == 1)
        self.assertEqual(top.digits, "29201")

    def test_two_numbers_on_one_paper_merge_into_one_document(self):
        # A single sheet showing the account number and, just below it, a branch number.
        # Close together -> one document, represented by the labeled account.
        account = acc("123456", 300, 100, 420, 130, label="מספר חשבון")
        branch = acc("7890", 300, 140, 380, 170)
        docs = group_documents([account, branch], self.s)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].digits, "123456")           # label-anchored read wins

    def test_three_papers_ranked_1_2_3(self):
        docs = group_documents([
            acc("500", 1200, 100, 1320, 130),
            acc("100", 100, 100, 220, 130),
            acc("300", 650, 100, 770, 130),
        ], self.s)
        self.assertEqual([d.digits for d in sorted(docs, key=lambda d: d.stack_position)],
                         ["100", "300", "500"])
        self.assertEqual([d.stack_position for d in
                          sorted(docs, key=lambda d: d.stack_position)], [1, 2, 3])

    def test_pages_tie_each_number_to_its_page(self):
        # Two detected pages; one account sits on each -> two documents, page_quad set,
        # ordered by number regardless of which page is left/right.
        right = acc("29200", 830, 300, 950, 340)
        left = acc("29201", 150, 300, 270, 340)
        page_right = [(760, 200), (1180, 200), (1180, 620), (760, 620)]
        page_left = [(80, 200), (500, 200), (500, 620), (80, 620)]
        docs = group_documents([right, left], self.s, pages=[page_right, page_left])
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(d.page_quad is not None for d in docs))
        top = next(d for d in docs if d.stack_position == 1)
        self.assertEqual(top.digits, "29200")
        self.assertEqual(top.page_quad, page_right)     # arrow will point at this page

    def test_number_off_all_pages_falls_back_to_cluster(self):
        # One number on a page, one stray number on no page -> still two documents.
        on_page = acc("29200", 830, 300, 950, 340)
        stray = acc("29999", 150, 900, 270, 940)
        page_right = [(760, 200), (1180, 200), (1180, 620), (760, 620)]
        docs = group_documents([on_page, stray], self.s, pages=[page_right])
        self.assertEqual(len(docs), 2)
        by_digits = {d.digits: d for d in docs}
        self.assertIsNotNone(by_digits["29200"].page_quad)   # tied to the page
        self.assertIsNone(by_digits["29999"].page_quad)      # clustered fallback

    def test_single_document_has_no_instruction(self):
        docs = group_documents([acc("29200", 100, 100, 220, 130)], self.s)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].stack_position, 1)
        self.assertEqual(stack_instruction(docs), "")        # nothing to stack

    def test_empty(self):
        self.assertEqual(group_documents([], self.s), [])


if __name__ == "__main__":
    unittest.main()
