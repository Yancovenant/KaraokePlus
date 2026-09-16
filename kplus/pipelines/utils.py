from __future__ import annotations

import re
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
    "overlap",
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

    # I never use this
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

    def to_line_idx(self, reference: str) -> ASRResult:
        lines = [line.strip() for line in reference.split("\n") if line.strip() and not line.startswith('[')]
        clusters, i = [], 0
        n = [len(line.split()) for line in lines]
        words = [w for text in self.texts for w in text.words]
        assert len(words) == sum(n)
        for l in n:
            clusters.append(words[i: i + l])
            i += l
        assert len(clusters) == len(lines)
        new_group_texts = [TextTiming(words=cluster) for cluster in clusters]
        self.texts = new_group_texts
        return self

    ASS_STYLE: t.ClassVar[list[str]] = [
        "Style: Lat_Duet,Montserrat Bold,120,&H0000A5FF&,&H00FFFFFF&,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1",
        "Style: CJK_Duet,Noto Sans CJK SC,120,&H0000A5FF&,&H00FFFFFF&,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1",
    ]
    
    ASS_HEADER: t.ClassVar[str] =  (
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

    def populate_ass(self) -> ASRResult:
        prev_end = 0.0
        n_segments = len(self.texts)
        for i, current in enumerate(self.texts):
            if not current.words: continue
            # Pad calculation
            pad_start = max(0.0, current.start - 0.8, prev_end)
            if i < n_segments - 1:
                after_start = self.texts[i+1].start
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
        return self


RE_CJK = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+')


def sec2ass(s: float) -> str:
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:0>1.0f}:{m:0>2.0f}:{s:0>2.2f}'


def overlap(a, b):
    return not (a.end < b.start or a.start > b.end)