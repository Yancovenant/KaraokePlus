import difflib
import re
import string
from functools import cached_property, lru_cache

from kplus import env

env.sequence_align, env.pypinyin, env.pykakasi, env.anyascii, env.jellyfish  # noqa: B018
# Need to be below
import jellyfish  # type: ignore
from anyascii import anyascii  # type: ignore
from pykakasi import kakasi  # type: ignore
from pypinyin import Style, pinyin  # type: ignore

kks = kakasi()

__all__ = [
    "RomajiPhonetic",
    "get_phonetic",
    "normalizekaldi",
    "safepath",
    "similarity",
    "token_similarity"
]

####################
# Text Normalization
####################
def safepath(s: str) -> str:
    return "".join([c for c in s if c.isalpha() or c.isdigit() or c in ' _-']).strip()

_PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)
def normalizekaldi(s:str) -> str:
    s = s.translate(_PUNCTUATION_TRANSLATOR).lower().strip()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s) # Kaldi
    return s

class RomajiPhonetic:
    def __init__(self, text: str):
        self.orig = text

    @cached_property
    def latin(self) -> str:
        """Convert CJK characters to their Romaji phonetic representation."""
        if any(0x3040 <= ord(c) <= 0x30FF for c in self.orig): # kana Japanese
            romanized = "".join([item['hepburn'] for item in kks.convert(self.orig)])
        elif any(0x4E00 <= ord(c) <= 0x9FFF for c in self.orig): # Han chinese
            pinyin_list = pinyin(self.orig, style=Style.NORMAL)
            romanized = "".join([item[0] for item in pinyin_list])
        else:
            romanized = anyascii(self.orig)
        clean_latin = "".join(ch for ch in romanized if ch.isalpha() or ch.isspace())
        return clean_latin.lower()

    @cached_property
    def phonetic(self) -> str:
        return jellyfish.metaphone(self.latin)

    def __eq__(self, other):
        return jellyfish.jaro_winkler_similarity(self.phonetic, other.phonetic) >= 0.5

@lru_cache(maxsize=2048)
def get_phonetic(word: str) -> RomajiPhonetic:
    return RomajiPhonetic(word)

def similarity(left: str, right: str) -> float:
    left = get_phonetic(left).latin
    right = get_phonetic(right).latin
    if not left or not right: return 0.0
    if left == right: return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()

def token_similarity(left: str, right: str) -> float:
    left_tokens = set(get_phonetic(left).latin.split())
    right_tokens = set(get_phonetic(right).latin.split())
    if not left_tokens: return 0.0
    intersection = (left_tokens & right_tokens)
    return len(intersection) / len(left_tokens)
