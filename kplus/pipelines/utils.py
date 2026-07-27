from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, TypeAlias

import kplus

if TYPE_CHECKING:
    import numpy as np, torch # type: ignore  # noqa: I001
    AudioType : TypeAlias = "torch.Tensor | np.ndarray | str"


def load_audio(audio_path: str, sr: float, channels: int) -> torch.Tensor:
    kplus.env.demucs  # noqa: B018
    from demucs.audio import AudioFile  # type: ignore
    return AudioFile(str(audio_path)).read(
        streams=0, samplerate=sr, channels=channels
    )

def convert_audio(audio: torch.Tensor, fromsr: float, tosr: float, channels=int) -> torch.Tensor:
    kplus.env.demucs  # noqa: B018
    from demucs.audio import convert_audio as julius_resampler  # type: ignore
    return julius_resampler(audio, fromsr, tosr, channels)


class TimingMixin:
    @property
    def duration(self) -> float:
        """Returns the length of the segment in seconds."""
        if self.start is None or self.end is None: 
            return 0.0
        return self.end - self.start

    def start(self): raise NotImplementedError()
    def end(self): raise NotImplementedError()

    def _to_hms(self, seconds: float | None) -> str:
        """Converts float seconds to MM:SS.ms format (e.g., 00:01.00)."""
        if seconds is None: 
            return "--:--.--"
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"

    @property
    def h_start(self) -> str:
        return self._to_hms(self.start)

    @property
    def h_end(self) -> str:
        return self._to_hms(self.end)


@dataclass(slots=True)
class AudioSegment(TimingMixin):
    start: float
    end: float

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, AudioSegment):
            return False
        return self.start == other.start and self.end == other.end


@dataclass
class WordTiming(TimingMixin):
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


@dataclass(slots=True)
class Segment(TimingMixin):
    words: list[WordTiming]
    language: str

    @property
    def text(self) -> str:
        return " ".join([w.word for w in self.words])

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


@dataclass(slots=True)
class Result:
    segments: list[Segment]

    def to_lyrics_segment(self):
        new_segments = []
        all_words = [w for segs in self.segments for w in segs.words]
        for idx, group in groupby(all_words, key=lambda x: x.line_idx):
            words = list(group)
            new_segments.append(Segment(words=words))
        self.segments = new_segments
        return self
