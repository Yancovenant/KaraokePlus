from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby, pairwise
from pathlib import Path
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


def _process_audio(audio: AudioType, from_sr: int | None, to_sr: int | None) -> np.ndarray:
    kplus.env.numpy, kplus.env.torch, kplus.env.ffmpeg  # noqa: B018
    import numpy as np, torch  # type: ignore  # noqa: I001
    if isinstance(audio, torch.Tensor):
        assert from_sr is not None, "Passing ``torch.Tensor`` require to also have ``sr`` included"
        audio = convert_audio(audio, from_sr, to_sr, 1)
    elif isinstance(audio, (str, Path)):
        audio: torch.Tensor = load_audio(audio, to_sr, 1)
    if not isinstance(audio, np.ndarray):
        audio = audio.detach().cpu().numpy().squeeze()
    return audio


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
    language: str | None = None
    ass_event: str | None = None

    @property
    def text(self) -> str:
        return " ".join([w.word for w in self.words])

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


import re
from typing import ClassVar

RE_CJK = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+')

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

    
    ASS_STYLE: ClassVar[list[str]] = [
            "Style: Lat_Duet,Montserrat Bold,120,&H0000A5FF&,&H00FFFFFF&,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1",
            "Style: CJK_Duet,Noto Sans CJK SC,120,&H0000A5FF&,&H00FFFFFF&,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1",
        ]

    ASS_HEADER: ClassVar[str] =  (
            "[Script Info]\n"
            "Title: KaraokePlus+\nScriptType: v4.00+\n"
            "PlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, "
            "SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
            "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"{chr(10).join(ASS_STYLE)}" + "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    def populate_ass(self):
        prev_end = 0.0
        n_segments = len(self.segments)
        for i, current in enumerate(self.segments):
            if not current.words: continue
            # Pad calculation
            pad_start = max(0.0, current.start - 0.8, prev_end)
            if i < n_segments - 1:
                after_start = self.segments[i+1].start
                gap2next = max(0.0, after_start - current.end)
                pad_end = current.end + min(gap2next * 0.7, 1.5) # max 1.5s
            else: # Last segment just do add 1s
                pad_end = current.end + 1.0
            prev_end = pad_end
            # Effect
            fade_in_ms = max(0, min(300, int((current.start - pad_start) * 1000)))
            fade_out_ms = max(0, min(300, int((pad_end - current.end) * 1000)))
            is_cjk = bool(RE_CJK.search(current.text))
            style = "CJK_Duet" if is_cjk else "Lat_Duet"
            # Word Token
            k_tokens = []
            for w_idx, word in enumerate(current.words):
                if w_idx < len(current.words) - 1:
                    gap2nextword = max(0.0, current.words[w_idx + 1].start - word.end)
                else: # Last word
                    gap2nextword = max(0.0, pad_end - word.end)
                if w_idx == 0:
                    prev_word_ts = pad_start
                else:
                    prev_word_ts = current.words[w_idx - 1].end
                wait_end_sec = word.end + min(gap2nextword * 0.7, 0.6)
                wait_start_sec = max(word.start - 0.3, prev_word_ts)
                gap_start_sec = max(0.0, word.start - wait_start_sec)
                gap_end_sec = max(0.0, wait_end_sec - word.end)
                dur_start_cs = max(0, round(gap_start_sec * 100))
                dur_end_cs = max(0, round(gap_end_sec * 100))
                dur_sec = max(0.0, word.end - word.start)
                dur_cs = max(0, round(dur_sec * 100))
                k_tokens.append(
                    f"{{\\kf{dur_start_cs}}}"
                    f"{{\\kf{dur_cs}}}{word.word.strip()}"
                    f"{{\\kf{dur_end_cs}}} "
                )
            karaoke_content = "".join(k_tokens)
            current.ass_event = (
                f"Dialogue: 0,{sec2ass(pad_start)},{sec2ass(pad_end)},"
                f"{style},,0,0,0,,"
                f"{{\\fad({fade_in_ms},{fade_out_ms})}}"
                f"{{\\an2}}{karaoke_content}"
            )


def sec2ass(s: (float, int)) -> str:
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:0>1.0f}:{m:0>2.0f}:{s:0>2.2f}'