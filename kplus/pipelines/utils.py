from __future__ import annotations

import typing as t
from dataclasses import dataclass

from kplus.tools import rich
from kplus.tools.audio import Audio, _HumanTime, _TimingMixin
from kplus.tools.text import RomajiPhonetic, get_phonetic

if t.TYPE_CHECKING:
    from kplus.tools.audio import AudioNumpy

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

    _phone: RomajiPhonetic | ... = ...
    
    @property
    def phone(self) -> RomajiPhonetic:
        if self._phone is not ...: return self._phone
        self._phone = get_phonetic(self.word)
        return self._phone
        
    @property
    def latin(self) -> str:
        return self.phone.latin


@dataclass(slots=True)
class TextTiming(_HumanTime):
    words: list[WordTiming]
    
    language: str | None = None
    ass_event: str | None = None

    _text: str | ... = ...
    _latin: str | ... = ...
    
    @property
    def text(self) -> str:
        if self._text is not ...: return self._text
        self._text = " ".join([w.word for w in self.words])
        return self._text

    @property
    def latin(self) -> str:
        if self._latin is not ...: return self._latin
        self._latin = " ".join([w.latin for w in self.words])
        return self._latin

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

    def display_audio(self, audio: AudioNumpy, sr: int, *, offset: float = 0.0) -> None:
        for w in self.words:
            rich.print(f"[{w.starth}-{w.endh}] ({w.duration:.3f}) - {w.word}")
            if w.start is None or w.end is None: continue
            audio_chunk = Audio.slicenp(audio, w.start - offset, w.end - offset, sr)
            if audio_chunk.shape[0] > 0:
                Audio.display_audio(audio_chunk, sr=sr)
            else:
                rich.print("~No Audio~")
            del audio_chunk

@dataclass(slots=True)
class ASRResult:
    texts: list[TextTiming]
