from functools import cached_property

def is_cjk_char(ch: str) -> bool:
    """Check if a character is a CJK (Chinese, Japanese, Korean) character."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF        # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF     # Extension A
        or 0x20000 <= code <= 0x2A6DF   # Extension B
        or 0x2A700 <= code <= 0x2B73F   # Extension C
        or 0x2B740 <= code <= 0x2B81F   # Extension D
        or 0x2B820 <= code <= 0x2CEAF   # Extension E
        or 0x2CEB0 <= code <= 0x2EBEF   # Extension F
        or 0x30000 <= code <= 0x3134F   # Extension G
        or 0x31350 <= code <= 0x323AF   # Extension H
        or 0x2EBF0 <= code <= 0x2EE5F   # Extension I
        or 0x3000 <= code <= 0x303F     # CJK Symbols and Punctuation
        or 0xF900 <= code <= 0xFAFF     # Compatibility Ideographs
        or 0x3040 <= code <= 0x309F     # Japanese Hiragana
        or 0x30A0 <= code <= 0x30FF     # Japanese Katanaka
        or 0x31F0 <= code <= 0x31FF     # Japanese Katanaka Phonetic Extension (Ainu)
        or 0xFF65 <= code <= 0xFF9F     # Japanese Legacy Katanaka (Narrow)
        or 0xAC00 <= code <= 0xD7AF     # Hangul Syllables (Modern Korean text)
        or 0x1100 <= code <= 0x11FF     # Hangul Jamo
        or 0x3130 <= code <= 0x318F     # Hangul Compatibility Jamo 
    )

# Japanese, Chinese, and Korean commonly known as CJK characters.
# - Japanese characters include Hiragana, Katakana, and Kanji.
# >> Kanji are Chinese borrowed characters
import jellyfish  # type: ignore
from anyascii import anyascii  # type: ignore
from pykakasi import kakasi  # type: ignore
from pypinyin import Style, pinyin  # type: ignore

kks = kakasi()
# - Korean characters include Hangul, Hanja
# >> Hanja are Chinese borrowed characters

class RomajiPhonetic:
    def __init__(self, text: str):
        self.orig = text

    @cached_property
    def latin(self):
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
    def phonetic(self):
        return jellyfish.metaphone(self.latin)

    def __eq__(self, other):
        return jellyfish.jaro_winkler_similarity(self.phonetic, other.phonetic) >= 0.75
    