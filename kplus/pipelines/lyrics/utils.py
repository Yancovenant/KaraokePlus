from __future__ import annotations

import typing as t
from collections import defaultdict
from dataclasses import dataclass, field

from kplus.pipelines.utils import TextTiming

if t.TYPE_CHECKING:
    from kplus.tools.text import RomajiPhonetic


class LyricAlignError(Exception):
    """ Error for lyrics alignment. """

@dataclass(slots=True)
class Token:
    word: str
    start: float | None
    end: float | None
    language: str | None
    line_idx: int | None
    score: float | None

    _clean: str | None = field(init=False, default=...)
    _phone: RomajiPhonetic | None = field(init=False, default=...)

    @property
    def clean(self) -> str:
        if self._clean is not ...:
            return self._clean
        from kplus.tools.text import normalizekaldi
        self._clean = normalizekaldi(self.word)
        return self._clean

    @property
    def phone(self) -> RomajiPhonetic:
        if self._phone is not ...:
            return self._phone
        from kplus.tools.text import get_phonetic
        self._phone = get_phonetic(self.word)
        return self._phone


@dataclass(slots=True)
class Tokens:
    tokens: list
    lines: list | None

    _cleans: list[str] | None = field(init=False, default=...)
    _groups: dict[int, list[Token]] | None = field(init=False, default=...)

    @classmethod
    def from_reference(cls, reference: str) -> Tokens:
        from kplus.tools.text import normalizekaldi
        reference = normalizekaldi(reference)
        lines = [
            line.strip()
            for line in reference.split("\n")
            if line.strip() and
            not line.startswith('[')
        ]
        tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            Token(
                word=word, score=None,
                start=None, end=None, 
                line_idx=i, language=None,
            )
            for i, line in enumerate(lines)
            for word in line.split()
            if normalizekaldi(word).strip()
        ]
        return cls(tokens=tokens, lines=lines)

    @classmethod
    def from_asr(cls, texts: list[TextTiming]) -> Tokens:
        tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            Token(
                word=w.word, score=w.score,
                start=w.start, end=w.end,
                line_idx=None, language=text.language,
            )
            for text in texts
            for w in text.words
            if w.score >= 0.1
        ]
        return cls(tokens=tokens, lines=None)

    @property
    def cleans(self):
        if self._cleans is not ...:
            return self._cleans
        self._cleans = [
            text.clean
            for text in self.tokens
        ]
        return self._cleans

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    @property
    def groups(self):
        if self._groups is not ...:
            return self._groups
        _groups = defaultdict(list)
        for token in self:
            _groups[token.line_idx].append(token)
        self._groups = _groups
        return self._groups

    @property
    def text(self):
        return " ".join([
            w.word
            for w in self.tokens
        ])
