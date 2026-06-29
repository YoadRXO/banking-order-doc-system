"""Hebrew text normalization, label matching, and account-number extraction.

Stdlib-only (uses `re` and `difflib`) so it is unit-testable without heavy deps.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

from .config import Settings

# Hebrew combining marks (niqqud, cantillation) and the geresh/gershayim symbols.
_HEBREW_DIACRITICS = re.compile("[֑-ֽֿׁ-ׇ]")
_GERESH = re.compile("[׳״'\"`׳״]")  # geresh, gershayim, ascii quotes
# Keep only Hebrew letters (U+05D0–U+05EA) and whitespace; drops digits, latin,
# punctuation and OCR noise like "|".
_NON_HEBREW = re.compile(r"[^א-ת\s]")
_NON_DIGIT = re.compile(r"\D+")
# A run of digits possibly broken by separators we treat as part of the number.
_NUMBER_RUN = re.compile(r"\d[\d \-/.]*\d|\d")

# Final (sofit) Hebrew letters folded to their base form so OCR/spelling variants
# compare equal (e.g. trailing ן vs נ).
_HEBREW_FINALS = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}


def normalize_hebrew(text: str) -> str:
    """Canonical form for robust comparison.

    Removes niqqud/geresh, keeps only Hebrew letters + spaces (drops digits, latin
    and OCR noise), and folds final letters to their base form.
    """
    if not text:
        return ""
    text = _HEBREW_DIACRITICS.sub("", text)
    text = _GERESH.sub("", text)
    text = _NON_HEBREW.sub("", text)
    text = "".join(_HEBREW_FINALS.get(ch, ch) for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_to_digits(text: str) -> str:
    """Keep only 0-9."""
    return _NON_DIGIT.sub("", text)


def extract_number_candidates(text: str, settings: Settings) -> List[str]:
    """Return digit strings found in `text` whose length passes the configured filter."""
    candidates: List[str] = []
    for match in _NUMBER_RUN.finditer(text):
        digits = strip_to_digits(match.group())
        if _valid_length(digits, settings):
            candidates.append(digits)
    return candidates


def _valid_length(digits: str, settings: Settings) -> bool:
    n = len(digits)
    if settings.exact_lengths:
        return n in settings.exact_lengths
    return settings.min_digits <= n <= settings.max_digits


def label_score(text: str, settings: Settings) -> float:
    """Fuzzy similarity (0..1) of `text` to the closest configured label keyword.

    Tries both reading orientations, because Tesseract may emit Hebrew in visual
    (reversed) order rather than logical order.
    """
    norm = normalize_hebrew(text)
    if not norm:
        return 0.0
    forms = (norm, norm[::-1])
    best = 0.0
    for keyword in settings.label_keywords:
        kn = normalize_hebrew(keyword)
        if not kn:
            continue
        for form in forms:
            ratio = SequenceMatcher(None, form, kn).ratio()
            # Reward substring containment (e.g. "יתרת חשבון" contains the keyword),
            # but only for keywords long enough to be distinctive — a 2-letter
            # abbreviation like "חן" would otherwise match inside unrelated words.
            if len(kn) >= 3 and kn in form:
                ratio = max(ratio, 0.9)
            best = max(best, ratio)
    return best


def has_anchor_token(text: str, settings: Settings) -> bool:
    """True if a strong account anchor word appears in `text`, in either reading
    orientation.

    Long anchors (e.g. "חשבון") match as a substring so they survive OCR noise
    that glues neighbouring tokens together. Short 2-letter abbreviations (e.g.
    "חן" / "מח") must match a *whole word* — otherwise they would fire inside
    unrelated words like "מחיר" (price) and tag random numbers.
    """
    norm = normalize_hebrew(text)
    if not norm:
        return False
    forms = (norm, norm[::-1])
    tokens = norm.split()
    for raw in settings.anchor_tokens:
        anchor = normalize_hebrew(raw)
        if not anchor:
            continue
        if len(anchor) <= 2:
            if anchor in tokens or anchor[::-1] in tokens:
                return True
        elif any(anchor in form for form in forms):
            return True
    return False


def is_label(text: str, settings: Settings) -> bool:
    """Decide whether an OCR box is a bank-account label (box-level; used for debug).

    Detection itself uses the line-level `line_label_match` below, which requires the
    whole phrase (number word + חשבון) on one line.
    """
    return (has_anchor_token(text, settings)
            or contains_account_word(text, settings)
            or label_score(text, settings) >= settings.label_fuzzy_threshold)


def _tokens_from_text(text: str) -> List[str]:
    """Normalized word-tokens of `text`, in both reading orientations (Tesseract may
    emit Hebrew visually reversed)."""
    norm = normalize_hebrew(text)
    if not norm:
        return []
    return norm.split() + norm[::-1].split()


def line_token_set(texts) -> set:
    """All normalized word-tokens present on a line (union over its OCR boxes)."""
    tokens = set()
    for text in texts:
        tokens.update(_tokens_from_text(text))
    return tokens


def _token_matches(want: str, tokens, settings: Settings, threshold: float = None) -> bool:
    """Is `want` present among `tokens`? Short (<=2 char) words must match exactly;
    longer words allow OCR-noise tolerance (containment or fuzzy ratio)."""
    thr = settings.label_fuzzy_threshold if threshold is None else threshold
    for tok in tokens:
        if len(want) <= 2:
            if tok == want:
                return True
        elif want in tok or SequenceMatcher(None, tok, want).ratio() >= thr:
            return True
    return False


def line_label_match(texts, settings: Settings):
    """Return the accepted label phrase that this line matches, else None.

    A phrase matches when its *distinctive* word — the account noun, e.g. "חשבון",
    which is the longest word of the phrase — appears on the line. Requiring only
    this (not every word) keeps detection robust on live video, where OCR routinely
    drops the short "מספר"/"מס", while still rejecting branch (סניף) and savings
    (חסכון) lines because their distinctive word is different. Set
    `require_number_word=True` to additionally demand the "מספר"/"מס" word.
    """
    tokens = line_token_set(texts)
    if not tokens:
        return None
    for keyword in settings.label_keywords:
        normalized = normalize_hebrew(keyword)
        if not normalized:
            continue
        words = normalized.split()
        if not words:
            continue
        key = max(words, key=len)  # the account noun (e.g. "חשבון")
        if _token_matches(key, tokens, settings, settings.account_word_threshold):
            return keyword
    return None


def line_has_number_word(texts, settings: Settings) -> bool:
    """True if a 'number' word (מספר / מס …) appears on this line."""
    tokens = line_token_set(texts)
    return any(_token_matches(normalize_hebrew(w), tokens, settings)
               for w in settings.number_words if normalize_hebrew(w))


def contains_account_word(text: str, settings: Settings) -> bool:
    """True if this box holds the account noun (חשבון), in either orientation. Used to
    locate the label on a line so the number after it (its value side) can be taken."""
    want = normalize_hebrew(settings.account_word)
    return bool(want) and _token_matches(want, set(_tokens_from_text(text)), settings,
                                         settings.account_word_threshold)
