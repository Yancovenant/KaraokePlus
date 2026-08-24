from __future__ import annotations  # noqa: I001

from dataclasses import dataclass

import kplus.init  # noqa: F401
from kplus import env
from kplus.tools import rich
from kplus.tools.audio import Audio, AudioNumpy, AudioType

from kplus.pipelines.audio.plotter import AudioPlotter

# Need to be below
import numpy as np


@dataclass(slots=True, frozen=True)
class _Result:
    _name: str

    data: np.ndarray
    smoothed: np.ndarray
    mask: np.ndarray
    times: np.ndarray
    threshold: np.ndarray

    def plot(self, plotter: AudioPlotter, row: int = 1) -> None:
        plotter.scatter(row=2, x=self.times, y=self.smoothed, name=self._name)


class RmsResult(_Result):
    _name = "RMS"

class FluxResult(_Result):
    pass

class MelResult(_Result):
    pass


@dataclass(slots=True, frozen=True)
class AudioSegments:
    pass

@dataclass(slots=True, frozen=True)
class DetectionResult:
    times: np.ndarray
    rms: RmsResult
    flux: FluxResult
    mel: MelResult



class AudioExtractor:
    def __init__(self,
        precision_ms: float = 10,
        overlap: float = 0.75,
        use_filter: bool = True,
        **kwargs
    ) -> None:
        self.precision_ms = precision_ms
        self.overlap = overlap
        self.use_filter = use_filter
        self.scipy = env.scipy
        self.librosa = env.librosa
        if env.verbose:
            self.plotter = AudioPlotter(shared_xaxes=True, vertical_spacing=0.05, **kwargs)

    def frame_and_hop(self, sr: int) -> tuple[int, int]:
        hoplength = int((sr / 1000) * self.precision_ms) # if sr == 44100 and precision_ms == 1, hop_length = 44 samples
        framelength = round(hoplength / (1 - self.overlap)) # 150% of hop_length = 66 samples
        return hoplength, framelength

    def _preprocess_audio(self, audio: AudioType, sr: int) -> AudioNumpy:
        audionp = audio
        if not isinstance(audio, AudioNumpy):
            audionp = Audio(audio, samplerate=sr, channels=1).numpy
        assert isinstance(audionp, AudioNumpy)
        if self.use_filter:
            sos = self.scipy.signal.butter(10, [200, 5000], btype='bandpass', fs=sr, output='sos')
            audionp = self.scipy.signal.sosfilt(sos, audionp).astype(np.float32)
        return audionp

    def detect_rms(self, audio: AudioType, sr: int) -> RmsResult:
        audionp = self._preprocess_audio(audio, sr)
        hoplength, framelength = self.frame_and_hop(sr)
        rms = self.librosa.feature.rms(y=audionp, frame_length=framelength, hop_length=hoplength)[0]
        times = self.librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hoplength)
        uniform_f_length_sec = 0.060 #500ms original
        frames_per_half_sec = max(1, int(uniform_f_length_sec / (self.precision_ms / 1000)))
        rms_smoothed = self.scipy.ndimage.uniform_filter1d(rms, size=frames_per_half_sec)
        rms_noise_floor = np.percentile(rms_smoothed, 5)
        rms_threshold = rms_noise_floor + (np.std(rms_smoothed) * 0.2)
        rms_mask = (rms_smoothed > rms_threshold)
        result = RmsResult(
            data=rms,
            smoothed=rms_smoothed,
            mask=rms_mask,
            times=times,
            threshold=rms_threshold,
        )
        if env.verbose: result.plot(self.plotter)
        return result

    def detect_mel(self, audio: AudioType, sr: int) -> MelResult:
        audionp = self._preprocess_audio(audio, sr)

    def detect_flux(self, audio: AudioType, sr: int) -> FluxResult:
        audionp = self._preprocess_audio(audio, sr)

    def detect_all(self, audio: AudioType, sr: int) -> DetectionResult:
        audionp = self._preprocess_audio(audio, sr)
        rms = self.detect_rms(audionp, sr)
        flux = self.detect_flux(audionp, sr)
        mel = self.detect_mel(audionp, sr)
        audio_segments = self.get_audio_segments(audionp, sr)
        times = None
        return DetectionResult(
            times=times,
            rms=rms,
            flux=flux,
            mel=mel,
            audio_segments=audio_segments,
        )


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    path, sr = args[0], int(args[1])
    AudioExtractor().detect_all(path, sr)