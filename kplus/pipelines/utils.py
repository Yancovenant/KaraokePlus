from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias
import warnings

import kplus

kplus.env.sequence_align, kplus.env.pypinyin, kplus.env.pykakasi, kplus.env.anyascii, kplus.env.jellyfish  # noqa: B018
# Need to be below this line
from kplus.tools.romaji_converter import RomajiPhonetic

if TYPE_CHECKING:
    import numpy as np, torch # type: ignore  # noqa: I001
    AudioType : TypeAlias = "torch.Tensor | np.ndarray | str"
    AudioTensor : TypeAlias = "torch.Tensor"
        

@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)

def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


class AudioLoader:
    def __init__(self, audio: AudioType, **kwargs):
        self.audio_path : str | None = None
        self._process_audio(audio, **kwargs)

    @cached_property
    def info(self):
        stdout_data = subprocess.check_output([
            'ffprobe', "-loglevel", "panic",
            str(self.audio_path), '-print_format', 'json', '-show_format', '-show_streams'
        ])
        return json.loads(stdout_data.decode('utf-8'))

    def __repr__(self):
        features = [("path", self.audio_path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioLoader({features_str})"

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    @property
    def _audio_streams(self):
        return [index for index, stream in enumerate(self.info["streams"])
                if stream["codec_type"] == "audio"]

    def __len__(self):
        return len(self._audio_streams)

    def _read_audio(self, seek_time=None, duration=None,
                    streams=slice(None), samplerate=None,
                    channels=None):
        kplus.env.ffmpeg, kplus.env.numpy, kplus.env.torch # noqa: B018
        import torch, numpy as np  # type: ignore  # noqa: I001
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]
        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))
        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.audio_path)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]

            subprocess.run(command, check=True)
            wavs = []
            for stream, filename in zip(streams, filenames):
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels(stream)).t()
                if channels is not None:
                    wav = convert_audio_channels(wav, channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav

    def _process_audio(self, audio: AudioType, **kwargs) -> np.ndarray:
        kplus.env.numpy, kplus.env.torch # noqa: B018
        import torch, numpy as np  # type: ignore  # noqa: I001
        if isinstance(audio, (str, Path)):
            self.audio_path = str(audio)
            streams = kwargs.pop("streams", 0)
            self.audio_tensor = self._read_audio(streams=streams, **kwargs)
        elif isinstance(audio, torch.Tensor):
            self.audio_tensor = audio
        elif isinstance(audio, np.ndarray):
            self.audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        self.audio_np = self.audio_tensor.detach().cpu().numpy().squeeze()


def load_audio(audio_path: str, sr: float, channels: int, return_sr: bool = False) -> torch.Tensor:
    kplus.env.demucs  # noqa: B018
    from demucs.audio import AudioFile  # type: ignore
    audio_file = AudioFile(str(audio_path))
    if return_sr:
        return audio_file.read(streams=0, samplerate=sr, channels=channels), audio_file.samplerate()
    return audio_file.read(
        streams=0, samplerate=sr, channels=channels
    )

def convert_audio(audio: torch.Tensor, fromsr: float, tosr: float, channels=int) -> torch.Tensor:
    kplus.env.demucs  # noqa: B018
    from demucs.audio import convert_audio as julius_resampler  # type: ignore
    return julius_resampler(audio, fromsr, tosr, channels)

def _process_audio(audio: AudioType, from_sr: int | None, to_sr: int | None, return_sr: bool = False) -> np.ndarray:
    kplus.env.numpy, kplus.env.torch, kplus.env.ffmpeg  # noqa: B018
    import numpy as np, torch  # type: ignore  # noqa: I001
    if isinstance(audio, torch.Tensor):
        assert from_sr is not None, "Passing ``torch.Tensor`` require to also have ``sr`` included"
        audio = convert_audio(audio, from_sr, to_sr, 1)
    elif isinstance(audio, (str, Path)):
        if from_sr is None or return_sr:
            audio, from_sr = load_audio(audio, to_sr, 1, return_sr=return_sr)
        else:
            audio: torch.Tensor = load_audio(audio, to_sr, 1)
    if not isinstance(audio, np.ndarray):
        audio = audio.detach().cpu().numpy().squeeze()
    if return_sr:
        return audio, from_sr
    return audio


class TimingMixin:
    @property
    def duration(self) -> float:
        """Returns the length of the segment in seconds."""
        if self.start is None or self.end is None: return 0.0
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
    _start: float = field(init=False, default=None, repr=False)
    _end: float = field(init=False, default=None, repr=False)
    score: float | None = None

    start: float | None = None
    end:  float | None = None

    def __setstate__(self, state):
        """Intercepts the object when it is being loaded from cache/pickle."""
        if 'start' in state and '_start' not in state:
            warnings.warn(
                f"Legacy WordTiming object loaded for word '{state.get('word')}'. This object is deprecated and requires re-initialization.",
                DeprecationWarning,
                stacklevel=2
            )
            state['_start'] = state.pop('start', None)
            state['_end'] = state.pop('end', None)
        self.__dict__.update(state)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, val):
        if val != self._start and self._end is not None and val is not None:
            print(f'{"Start":<5}: [{self.h_start} -> {self._to_hms(val)}] "{self.word}"')
        self._start = val

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        if val != self._end and self._end is not None and val is not None:
            print(f'{"End":<5}: [{self.h_end} -> {self._to_hms(val)}] "{self.word}"')
        self._end = val

    @property
    def latin(self):
        return RomajiPhonetic(self.word.strip()).latin

@dataclass(slots=True)
class Segment(TimingMixin):
    words: list[WordTiming]
    language: str | list[str] | None = None
    ass_event: str | None = None

    @property
    def text(self) -> str:
        return " ".join([w.word for w in self.words])

    @property
    def latin(self) -> str:
        return " ".join(w.latin for w in self.words)

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
        return self


def sec2ass(s: (float, int)) -> str:
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:0>1.0f}:{m:0>2.0f}:{s:0>2.2f}'


