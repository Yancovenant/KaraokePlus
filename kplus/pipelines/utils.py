from __future__ import annotations

from dataclasses import dataclass

from kplus.tools.audio import _HumanTime, _TimingMixin

__all__ = [
    "ASRResult",
    "AudioSegment",
    "TextTiming",
    "WordTiming",
]

class AudioSegment(_TimingMixin):
    """ Responsible to hold audio segment """
    def __hash__(self):
        return hash((self.start, self.end))
    
    def __eq__(self, other):
        if not isinstance(other, AudioSegment):
            return False
        return self.start == other.start and self.end == other.end

@dataclass(slots=True)
class WordTiming(_TimingMixin):
    word: str
    score: float


@dataclass(slots=True)
class TextTiming(_HumanTime):
    words: list[WordTiming]
    
    language: str | None = None
    ass_event: str | None = None

    _text: str | ... = ...
    
    @property
    def text(self) -> str:
        if self._text is not ...: return self._text
        self._text = " ".join([w.word for w in self.words])
        return self._text

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None: return 0.0
        return float(round(self.end - self.start, 2))


@dataclass(slots=True)
class ASRResult:
    texts: list[TextTiming]
