from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass

from kplus.tools import rich
from kplus.tools.audio import Audio, _HumanTime, _TimingMixin
from kplus.tools.text import RomajiPhonetic, get_phonetic, normalizekaldi

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
        reference = normalizekaldi(reference)
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
        n_segments = len(self.texts)

        # Text Gap
        gaps = []
        for i in range(n_segments - 1):
            next_start = float(round(self.texts[i+1].start, 2))
            current_end = float(round(self.texts[i].end, 2))
            gap = round(next_start - current_end, 2)
            gaps.append(gap)
        assert len(gaps) == n_segments - 1

        # Populating Gaps
        sliced_gaps = []
        for gap in gaps:
            start_gap = round(gap * 0.3, 2)
            end_gap = round(gap - start_gap, 2)
            sliced_gaps.append((start_gap, end_gap))
        
        for i, text in enumerate(self.texts):
            if i == n_segments - 1:
                pad_end = 1.0 # just add 1.0s
            else:
                pad_end = min(sliced_gaps[i][0], 1.0)
            padded_end = round(text.end + pad_end, 2)
            if i == 0:
                pad_start = 0.8
            else:
                pad_start = min(sliced_gaps[i-1][1], 0.8)
            padded_start = max(0.0, round(text.start - pad_start, 2))

            # Effect
            fade_in_ms = max(0, min(300, int((text.start - padded_start) * 1000))) # 300ms max to show text before highlighting
            fade_out_ms = max(0, min(300, int((padded_end - text.end) * 1000))) # 300ms max to hide text before next word

            # TODO: Find better way
            is_cjk = bool(RE_CJK.search(text.text))
            style = "CJK_Duet" if is_cjk else "Lat_Duet"
            
            # Karaoke Content
            k_tokens = []
            total_word_dur = 0
            for j, word in enumerate(text.words):
                dur_cs = max(0.0, round(word.duration * 100))
                tokens = f"{{\\kf{dur_cs}}}{word.word.strip()} "
                if j == 0:
                    prev_gap = word.start - padded_start
                else:
                    prev_gap = word.start - text.words[j-1].end

                dur_start_cs = max(0.0, round(prev_gap * 100))
                tokens = f"{{\\kf{dur_start_cs}}}" + tokens
                total_dur = dur_cs + dur_start_cs

                if j == len(text.words) - 1:
                    next_gap = padded_end - word.end
                    dur_end_cs = max(0.0, round(next_gap * 100))
                    tokens = tokens + f"{{\\kf{dur_end_cs}}}"
                    total_dur += dur_end_cs
                
                k_tokens.append(tokens)
                total_word_dur += total_dur
            assert total_word_dur == round((padded_end - padded_start) * 100)

            karaoke_content = "".join(k_tokens)
            text.ass_event = (
                f"Dialogue: 0,{sec2ass(padded_start)},{sec2ass(padded_end)},"
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