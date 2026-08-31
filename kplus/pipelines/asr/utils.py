from __future__ import annotations  # noqa: I001

from kplus.tools.audio import _HumanTime, _TimingMixin

# Need to be below
import torch

from dataclasses import dataclass

__all__ = [
    "ASRResult",
    "TextTiming",
    "WordTiming",
    "get_default_dtype",
]

def get_default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


@dataclass(slots=True)
class WordTiming(_TimingMixin):
    word: str
    score: float


@dataclass(slots=True)
class TextTiming(_HumanTime):
    words: list[WordTiming]
    language: str

    _text: str | ... = ...
    _start: float | ... = ...
    _end: float | ... = ...
    _duration: float | ... = ...
    
    @property
    def text(self) -> str:
        if self._text is not ...: return self._text
        self._text = " ".join([w.word for w in self.words])
        return self._text

    @property
    def start(self) -> float:
        if self._start is not None: return self._start
        self._start = self.words[0].start
        return self._start

    @property
    def end(self) -> float:
        if self._end is not ...: return self._end
        self._end = self.words[-1].end
        return self._end

    @property
    def duration(self) -> float:
        if self._duration is not ...: return self._duration
        self._duration = self.end - self.start
        return self._duration


@dataclass(slots=True)
class ASRResult:
    texts: list[TextTiming]
