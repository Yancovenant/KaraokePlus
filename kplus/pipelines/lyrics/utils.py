from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from kplus.pipelines.utils import TextTiming, WordTiming
from kplus.tools import rich
from kplus.tools.text import RomajiPhonetic, get_phonetic, normalizekaldi


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

    _clean: str | ... = ...
    _phone: RomajiPhonetic | ... = ...

    @property
    def clean(self) -> str:
        if self._clean is not ...: return self._clean
        self._clean = normalizekaldi(self.word)
        return self._clean

    @property
    def phone(self) -> RomajiPhonetic:
        if self._phone is not ...: return self._phone
        self._phone = get_phonetic(self.word)
        return self._phone


@dataclass(slots=True)
class Tokens:
    tokens: list
    lines: list | None

    _cleans: list[str] | ... = ...
    _groups: dict[int, list[Token]] | ... = ...
    
    @classmethod
    def from_reference(cls, reference: str) -> Tokens:
        lines = [line.strip() for line in reference.split("\n") if line.strip() and not line.startswith('[')]
        tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            Token(
                word=word, score=None,
                start=None, end=None, 
                line_idx=i, language=None,
            ) for i, line in enumerate(lines)
            for word in line.split()
        ]
        return cls(tokens=tokens, lines=lines)

    @classmethod
    def from_asr(cls, texts: list[TextTiming]) -> Tokens:
        tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            Token(
                word=w.word, score=w.score,
                start=w.start, end=w.end,
                line_idx=None, language=t.language,
            ) for t in texts for w in t.words if w.score >= 0.1
        ]
        return cls(tokens=tokens, lines=None)

    @property
    def cleans(self):
        if self._cleans is not ...: return self._cleans
        self._cleans = [t.clean for t in self.tokens]
        return self._cleans

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    @property
    def groups(self):
        if self._groups is not ...: return self._groups
        _groups = defaultdict(list)
        for token in self:
            _groups[token.line_idx].append(token)
        self._groups = _groups
        return self._groups


@dataclass(slots=True)
class AudioAlignment:
    line_idx: int
    tokens: list[Token] # per word
    audio_ids: list[int] = field(default_factory=list)

    
    def to_texttiming(self) -> TextTiming:
        return TextTiming(words=[
            WordTiming(
                start=t.start, end=t.end,
                score=t.score, word=t.word, 
            ) for t in self.tokens
        ], language=self.tokens[0].language)
