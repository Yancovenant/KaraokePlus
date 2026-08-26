from __future__ import annotations

# Should be imported only when needed to not make loading slower
from kplus import env

# Must be at the top
env.ffmpeg, env.torch, env.numpy  # noqa: B018

import base64
import io
import json
import subprocess
import typing as t
import wave
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from .path import temp_filenames

__all__ = [
    "Audio",
    "AudioNumpy",
    "AudioTensor",
    "AudioType",
    "_TimingMixin",
]

AudioNumpy: t.TypeAlias = np.ndarray
AudioTensor: t.TypeAlias = torch.Tensor
AudioType: t.TypeAlias = str | Path | AudioNumpy | AudioTensor

class Audio:
    """ Audio Loader and Manager """
    def __init__(self,
        audio: AudioType,
        samplerate: int | None = None,
        channels: int | None = None,
        **kwargs
    ) -> None:
        self.audiopath: str = ""
        self.init(audio, samplerate, channels, **kwargs)
        self.tensor: AudioTensor
        self.numpy: AudioNumpy

    def __repr__(self):
        features = [("path", self.audiopath)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioLoader({features_str})"

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])
    
    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    @cached_property
    def info(self):
        stdout_data = subprocess.check_output([
            'ffprobe', "-loglevel", "panic",
            str(self.audiopath), '-print_format', 'json', '-show_format', '-show_streams'
        ])
        return json.loads(stdout_data.decode('utf-8'))

    @cached_property
    def _audio_streams(self):
        return [index for index, stream in enumerate(self.info["streams"])
                if stream["codec_type"] == "audio"]

    def _read_audio(self,
        seek_time=None,
        duration=None,
        streams=slice(None),
        samplerate=None,
        channels=None
    ) -> AudioTensor:
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single: streams = [streams]
        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))
        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time: command += ['-ss', str(seek_time)]
            command += ['-i', str(self.audiopath)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None: command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None: command += ['-ar', str(samplerate)]
                command += [filename]
            subprocess.run(command, check=True)
            wavs = []
            for stream, filename in zip(streams, filenames):
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels(stream)).t()
                if channels is not None:
                    wav = self.convert_audio_channels(wav, channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav

    @staticmethod
    def np2base64(audio: AudioNumpy, sr: int) -> str:
        # Make mono
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        # Prevent clipping
        audio = np.clip(audio, -1.0, 1.0)
        # float32 -> signed 16-bit PCM
        pcm = (audio * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # int16
            wav.setframerate(sr)
            wav.writeframes(pcm.tobytes())
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    @cached_property
    def base64(self) -> str:
        """ Convert numpy audio to base64, require sr"""
        audio = np.asarray(self.numpy, dtype=np.float32)
        sr = self.samplerate()
        return self.np2base64(audio, sr)
        

    def init(self,
        audio: AudioType,
        samplerate: int | None = None,
        channels: int | None = None,
        **kwargs
    ) -> None:
        if isinstance(audio, (str, Path)):
            self.audiopath = str(Path(str(audio)).expanduser().resolve())
            streams = kwargs.pop("streams", 0)
            self.tensor = self._read_audio(samplerate=samplerate, channels=channels, streams=streams, **kwargs)
        else:
            if isinstance(audio, AudioTensor):
                self.tensor = audio
            elif isinstance(audio, AudioNumpy):
                self.tensor = torch.from_numpy(audio).unsqueeze(0)
            else:
                raise TypeError(f"Unsupported audio type: {type(audio)}")
        self.numpy = self.tensor.detach().cpu().numpy().squeeze()


    @staticmethod
    def convert_audio_channels(wav: AudioTensor, channels: int = 2) -> AudioTensor:
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


@dataclass(slots=True)
class _HumanTime:
    """ Helper Mixin for rendering human readable timing """

@dataclass(slots=True)
class _TimingMixin(_HumanTime):
    start: float
    end: float

    _duration: float | ... = field(default=..., init=False)
    @property
    def duration(self) -> float:
        if self._duration is not ...: return ...
        self._duration = self.end - self.start
        return self._duration