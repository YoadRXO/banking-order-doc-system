"""Group account detections into physical documents and compute the stacking order.

The multi-document feature: when two (or more) separate papers are shown at once, each
with its own bank account number, tell the user which paper to place on **TOP** of the
stack. The document whose account number is first in the sort order goes on top.

    Right paper 29200, left paper 29201  →  put 29200 on top, 29201 under it.

Note: which paper is on the left/right does NOT decide the order — the account *number*
does. Here the smaller number happens to be on the right, so the right paper goes on top.

Pure logic (stdlib + types only) so it is unit-tested without OpenCV / the OCR engine.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .config import Settings
from .types import AccountNumber, DocumentGroup, Point


def _gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Edge-to-edge gap between two axis-aligned boxes (0 if they touch/overlap).

    Returned as the larger of the horizontal and vertical gaps, so two boxes count as
    "close" only when they are near on BOTH axes.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return max(dx, dy)


def _cluster(accounts: List[AccountNumber], settings: Settings) -> List[List[AccountNumber]]:
    """Single-linkage clustering of account boxes into documents.

    Two numbers belong to the same document when the gap between their boxes is small
    relative to their size (`stack_gap_factor` × the larger box side). Numbers on two
    papers held apart (a clear gap between the sheets) fall into separate clusters.
    """
    n = len(accounts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        bi = accounts[i].detection
        for j in range(i + 1, n):
            bj = accounts[j].detection
            # Scale by text HEIGHT (a stable unit, like the rest of the spatial math) —
            # not width: account numbers are wide, so a width-based gap would merge two
            # papers held side by side. Numbers within `stack_gap_factor` line-heights on
            # both axes are the same document; papers held apart stay separate.
            ref = max(bi.height, bj.height, 1.0)
            if _gap(bi.bbox, bj.bbox) <= settings.stack_gap_factor * ref:
                union(i, j)

    clusters: dict = {}
    for idx, acc in enumerate(accounts):
        clusters.setdefault(find(idx), []).append(acc)
    return list(clusters.values())


def _quad_bbox(quad: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _assign_to_pages(accounts: List[AccountNumber], pages: List[List[Point]]):
    """Bucket each account under the tightest detected page whose box holds its centre.

    Returns (page_docs, leftover): one DocumentGroup per page that caught a number (with
    its `page_quad` set), plus the accounts that fell on no page (to be gap-clustered).
    """
    page_boxes = [(_quad_bbox(q), q) for q in pages]
    buckets: dict = {}
    leftover: List[AccountNumber] = []
    for acc in accounts:
        cx, cy = acc.detection.center
        best_pi, best_area = None, None
        for pi, (bb, _q) in enumerate(page_boxes):
            if bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
                area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                if best_area is None or area < best_area:  # tightest page wins
                    best_pi, best_area = pi, area
        if best_pi is None:
            leftover.append(acc)
        else:
            buckets.setdefault(best_pi, []).append(acc)
    page_docs = [DocumentGroup(accounts=accs, page_quad=page_boxes[pi][1])
                 for pi, accs in buckets.items()]
    return page_docs, leftover


def group_documents(accounts: List[AccountNumber], settings: Settings,
                    pages: Optional[List[List[Point]]] = None) -> List[DocumentGroup]:
    """Cluster `accounts` into documents and order them for stacking.

    When `pages` (detected paper rectangles) are given, each number is tied to the page it
    sits on; numbers on no detected page fall back to gap-clustering. Returns DocumentGroups
    sorted by stacking order, each with `stack_position` set (**1 = TOP of the stack**).
    Ordering follows `settings.ascending`: ascending means the smallest number goes on top.
    """
    if not accounts:
        return []
    if pages:
        docs, leftover = _assign_to_pages(accounts, pages)
    else:
        docs, leftover = [], accounts
    docs += [DocumentGroup(accounts=c) for c in _cluster(leftover, settings)]
    # Stable left-to-right tie-break, then order by account value (top = first in order).
    docs.sort(key=lambda d: d.region[0])
    docs.sort(key=lambda d: d.value, reverse=not settings.ascending)
    for position, doc in enumerate(docs, start=1):
        doc.stack_position = position
    return docs


def stack_instruction(docs: List[DocumentGroup]) -> str:
    """One plain-language line, e.g. 'Put 29200 on top, then 29201 under it.'

    Returns "" when there are fewer than two documents (nothing to stack).
    """
    ordered = sorted(docs, key=lambda d: d.stack_position or 0)
    if len(ordered) < 2:
        return ""
    nums = [d.digits for d in ordered]
    return f"Put {nums[0]} on top, then " + ", then ".join(nums[1:]) + " under it."
