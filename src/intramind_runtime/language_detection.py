"""Deterministic, offline prose identification; uncertain text stays unclassified.

langdetect ships its profiles in the installed wheel. No runtime download or LLM
is used. Each detector has its own state, created from an immutable seeded factory.
"""

import re
import unicodedata
from functools import lru_cache

_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_MASK = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|"
    r"https?://\S+|\[(?:(?:Nguồn|Source|S)\s*)?\d+(?:[ ,;-]+\d+)*\]|<[^>\n]+>|\d+"
)
_SCRIPT_TARGETS = (
    (re.compile(r"[\u3040-\u30ff]"), {"ja"}),
    (re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"), {"ko"}),
    (re.compile(r"[\u0400-\u052f]"), {"ru", "uk", "be", "bg", "mk", "sr", "kk", "ky", "mn", "tg"}),
    (re.compile(r"[\u0370-\u03ff]{3,}"), {"el"}),
    (re.compile(r"[\u0600-\u06ff]"), {"ar", "fa", "ur", "ps", "ug"}),
    (re.compile(r"[\u0590-\u05ff]"), {"he", "yi"}),
    (re.compile(r"[\u0900-\u097f]"), {"hi", "mr", "ne", "sa"}),
    (re.compile(r"[\u0e00-\u0e7f]"), {"th"}),
    (re.compile(r"[\u1780-\u17ff]"), {"km"}),
    (re.compile(r"[\u0e80-\u0eff]"), {"lo"}),
)


@lru_cache(maxsize=1)
def _factory():
    # Imported only from activities/direct callers, never during workflow import.
    from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory

    factory = DetectorFactory()
    factory.seed = 0
    factory.load_profile(PROFILES_DIRECTORY)
    return factory


def detect_language(text: str, *, minimum_letters=8, minimum_words=2, confidence=0.98):
    """Return a confident base language code, or None; never infer from IDs alone."""
    from langdetect.lang_detect_exception import LangDetectException

    candidate = _MASK.sub(" ", unicodedata.normalize("NFC", text))
    letters = sum(c.isalpha() for c in candidate)
    if letters < minimum_letters or len(_WORDS.findall(candidate)) < minimum_words:
        return None
    detector = _factory().create()
    detector.append(candidate[:4000])
    try:
        results = detector.get_probabilities()
    except LangDetectException:
        return None
    if not results or results[0].prob < confidence:
        return None
    return results[0].lang.split("-")[0]


def wrong_language(text: str, target: str) -> bool:
    """Require script evidence or confident prose evidence, not a lone foreign term."""
    candidate = _MASK.sub(" ", unicodedata.normalize("NFC", text))
    target = target.split("-")[0]
    # One Greek letter may be a formula; a Cyrillic/Thai word is stronger evidence.
    if any(
        target not in allowed and len(pattern.findall(candidate)) >= 3
        for pattern, allowed in _SCRIPT_TARGETS
    ):
        return True
    supported = {code.split("-")[0] for code in _factory().langlist}
    if target not in supported:
        return False
    # Check individual sentences as well as the whole value, so a correct-language
    # introduction cannot hide a paragraph generated in another language.
    parts = [candidate[:4000], *re.split(r"[.!?。！？\n]+", candidate)]
    if len(parts) > 25:
        parts = [parts[round(i * (len(parts) - 1) / 24)] for i in range(25)]
    for part in dict.fromkeys(parts):
        words = _WORDS.findall(part)
        unspaced = (
            sum(
                "CJK" in unicodedata.name(char, "")
                or "HIRAGANA" in unicodedata.name(char, "")
                or "KATAKANA" in unicodedata.name(char, "")
                for char in part
            )
            >= 20
        )
        if not unspaced and sum(word.islower() for word in words) < 4:
            continue  # e.g. a list of product names, rather than generated prose
        detected = detect_language(part, minimum_letters=40, minimum_words=0 if unspaced else 7)
        if detected and detected != target:
            return True
    return False
