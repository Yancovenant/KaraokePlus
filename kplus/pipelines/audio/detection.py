from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from kplus import env
from kplus.pipelines.audio.plotter import AudioPlotter
from kplus.pipelines.utils import AudioSegment

# Need to be below
env.numpy  # noqa: B018
import numpy as np

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from kplus.tools.audio import AudioNumpy, AudioType


#@dataclass(slots=True)
class ExtractorConfig:
    """ Configuration dictionary for extraction """
    smooth_distance_ms: int = 60 # 60ms

    peaks_prominence: float = 0.01 # 1% stands out of maximum given data

    floor_percentile: int = 5 # 5th order from lowest
    std_multiplier: float = 0.2 # Standard deviation multipler to be used to find the threshold floor

    max_forward_distance_ms: int = 200 # 200ms

    merge_ms: int = 140
    delete_ms: int = 100
    merge_lower_ms: int = 3000

    max_maskblock_ms: int = 10000 # 10s
    min_maskblock_ms: int = 1000 # 1s
    min_silence_ms: int = 80 # 10ms
    

class Extractor:
    """ Helper class to hold all computation of waveform signal """

    def __init__(self, precision_ms: int, **kwargs):
        self.precision_ms = precision_ms
        if env.verbose:
            self.plotter = AudioPlotter(
                shared_xaxes=True,
                vertical_spacing=0.05,
                **kwargs
            )

    @property
    def scipy(self):
        return env.scipy
        
    @property
    def librosa(self):
        return env.librosa

    def smooth_ms(self, data: np.ndarray, distance_ms: int) -> np.ndarray:
        framesize = max(1, int(distance_ms / self.precision_ms))
        if framesize % 2 == 0: framesize += 1 # Make it odd for centering
        return self.scipy.ndimage.uniform_filter1d(data, size=framesize)

    def peaks(self, data: np.ndarray, prominence: float) -> np.ndarray:
        return self.scipy.signal.find_peaks(data, prominence=prominence)[0]

    def mask_forward(
        self,
        datamask: np.ndarray,
        to: np.ndarray,
        max_distance_ms: int = ExtractorConfig.max_forward_distance_ms
    ) -> np.ndarray:
        edges = np.where((
            datamask[:-1] == True) & (datamask[1:] == False
        ))[0]
        maxframesize = int(max_distance_ms / self.precision_ms)
        for e in edges:
            maxdist = e + maxframesize
            after = to[to > e]
            if len(after) > 0:
                after = after[0]
                datamask[e:min(after, maxdist)] = True
        return datamask

    def ms2frame(self, ms: int) -> int:
        return max(1, int(ms / self.precision_ms)) # 140ms gap
        
    def get_mask_starts_ends(self, datamask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diffs = np.diff(np.pad(datamask.astype(int), pad_width=1, constant_values=0))
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0] 
        return starts, ends
    
    def mask_merge_lower_duration(
        self,
        datamask: np.ndarray,
        merge_ms: int = ExtractorConfig.merge_lower_ms
    ) -> np.ndarray:
        """ Any mask frame `lower than or equal` the threshold ms, would be merged to the nearest mask """

        while True:
            starts, ends = self.get_mask_starts_ends(datamask)
            mergeframe = self.ms2frame(merge_ms)
            if len(starts) == 0:
                break

            shortsegments = []
            for i, (s, e) in enumerate(zip(starts, ends)):
                dur = e - s
                if dur <= mergeframe:
                    shortsegments.append((i, s, e))
            if not shortsegments:
                break

            assert len(starts) > 1
            for i, s, e in shortsegments:
                next_start = starts[i+1] if i < len(starts) - 1 else None
                prev_end = ends[i-1] if i > 0 else None
                prev_gap = (s - prev_end) if prev_end is not None else float("inf")
                next_gap = (next_start - e) if next_start is not None else float("inf")
                if prev_gap < next_gap:
                    datamask[prev_end:s] = True
                elif next_gap < prev_gap:
                    datamask[e:next_start] = True
                elif next_gap == prev_gap:
                    next_end = ends[i+1] if i < len(ends) - 1 else None
                    next_dur = (next_end - next_start) if (next_start is not None and next_end is not None) else float("inf")
                    prev_start = starts[i-1] if i > 0 else None
                    prev_dur = (prev_end - prev_start) if (prev_start is not None and prev_end is not None) else float("inf")
                    if next_dur < prev_dur:
                        datamask[e:next_start] = True
                    else:
                        datamask[prev_end:s] = True
                else:
                    raise RuntimeError("Not sure why this happen")
        return datamask

    def mask_merge(self, datamask: np.ndarray, merge_ms: int = ExtractorConfig.merge_ms) -> np.ndarray:
        """ Any distance `lower than or equal` the threshold ms, would be merged """
        starts, ends = self.get_mask_starts_ends(datamask)
        mergeframe = self.ms2frame(merge_ms)
        for s, e in zip(ends[:-1], starts[1:]): # i.e. the end of one True block, the start of the next True block
            dist = e - s
            if dist <= mergeframe:
                datamask[s:e] = True
        return datamask

    def mask_delete(self, datamask: np.ndarray, delete_ms: int = ExtractorConfig.delete_ms) -> np.ndarray:
        """ Any mask length `lower than or equal` the threshold ms, would be deleted """
        starts, ends = self.get_mask_starts_ends(datamask)
        deleteframe = self.ms2frame(delete_ms)
        for s, e in zip(starts, ends):
            dur = e - s
            if dur <= deleteframe:
                datamask[s:e] = False
        return datamask


@dataclass(slots=True)
class Feature:
    """ Base Class for all extracted signal """
    _name: t.ClassVar[str]
    
    extractor: Extractor
    hoplength: int
    framelength: int
    sr: int
    
    data: np.ndarray
    
    _times: np.ndarray | None = field(init=False, default=...)
    _smoothed: np.ndarray | None = field(init=False, default=...)
    _valleys: np.ndarray | None = field(init=False, default=...)
    _threshold: np.float32 | None = field(init=False, default=...)
    _mask: np.ndarray | None = field(init=False, default=...)

    @property
    def times(self) -> np.ndarray:
        if self._times is not ...:
            return self._times

        self._times = (
            self.extractor.librosa
            .times_like(
                self.data,
                sr=self.sr,
                hop_length=self.hoplength
            )
        )
        return self._times

    @times.setter
    def times(self, value: np.ndarray) -> None:
        self._times = value

    @property
    def smoothed(self) -> np.ndarray:
        if self._smoothed is not ...:
            return self._smoothed
        self._smoothed = self.extractor.smooth_ms(
            self.data,
            ExtractorConfig.smooth_distance_ms
        )
        return self._smoothed

    def valley_times(self, safe_start: float = 0.0) -> np.ndarray:
        return self.times[self.valleys] + safe_start
        
    @property
    def valleys(self) -> np.ndarray:
        """ Local minima of the feature signal """
        if self._valleys is not ...:
            return self._valleys
        # Normalized
        _max = np.max(np.abs(self.smoothed))
        norm = (self.smoothed / _max) if _max > 0 else self.smoothed
        inverted = -norm
        self._valleys = self.extractor.peaks(inverted, ExtractorConfig.peaks_prominence)
        return self._valleys

    @property
    def threshold(self) -> np.float32:
        if self._threshold is not ...:
            return self._threshold
        noise_floor = np.percentile(self.smoothed, ExtractorConfig.floor_percentile)
        self._threshold = noise_floor + (np.std(self.smoothed) * ExtractorConfig.std_multiplier)
        return self._threshold

    def mask_times(self, safe_start: float = 0.0) -> tuple[float, float]:
        scipy = self.extractor.scipy
        find_objects, label = scipy.ndimage.find_objects, scipy.ndimage.label
        shifted_times = self.times + safe_start
        slices = [slc[0] for slc in find_objects(label(self.mask)[0])]
        return [
            (float(round(shifted_times[slc.start], 2)), float(round(shifted_times[slc.stop], 2)))
            for slc in slices
        ]
        
    @property
    def mask(self) -> np.ndarray:
        if self._mask is not ...:
            return self._mask
        self._mask = (self.smoothed > self.threshold)
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray) -> None:
        self._mask = value

    def plot(self, show=False, row: int = 1) -> None:
        plotter = self.extractor.plotter

        plotter.scatter(
            row=row, x=self.times, y=self.mask * np.max(self.smoothed),
            name=self._name + " Mask",
            fill="tozeroy", line=dict(shape="hv"),
        )
        plotter.scatter(
            row=row, x=self.times, y=self.smoothed,
            name=self._name + " Smoothed"
        )
        plotter.scatter(
            row=row, x=[self.times[0], self.times[-1], None], y=[self.threshold, self.threshold, None],
            name=self._name + " Threshold"
        )
        plotter.scatter(
            row=row, x=self.times[self.valleys], y=self.smoothed[self.valleys],
            name=self._name + " Valleys", mode="markers"
        )
        if show: plotter.show()


class Rms(Feature):
    """ Root Mean Square (e.g Waveform) """
    _name = "RMS"

class Flux(Feature):
    """ Accoustic strength """
    _name = "Flux"

@dataclass(slots=True)
class Mel(Feature):
    """ Log Mel Spectogram """
    _name = "Mel"

    S: np.ndarray

    _S_dB: np.ndarray | None = field(init=False, default=...)

    @property
    def S_dB(self) -> np.ndarray:
        if self._S_dB is not ...:
            return self._S_dB
        self._S_dB = self.extractor.librosa.power_to_db(self.S, ref=np.max)
        return self._S_dB

    def plot(self, show=False, row: int = 1) -> None:
        env.matplotlib  # noqa: B018
        import base64  # noqa: I001
        from io import BytesIO
        import matplotlib.pyplot as plt
        from PIL import Image
        plotter = self.extractor.plotter
        freq = self.extractor.librosa.mel_frequencies(n_mels=self.S.shape[0], fmax=5000)
        z = self.S_dB
        normalized = (z - z.min()) / (z.max() - z.min())  # scale between 0 and 1
        colormap = plt.get_cmap('magma')
        rgba_img = colormap(normalized)
        img_array = (rgba_img * 255).astype(np.uint8)
        
        image = Image.fromarray(img_array)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        png = buffer.getvalue()
        
        encoded_png = base64.b64encode(png).decode('utf-8')
        png_str = f"data:image/png;base64,{encoded_png}"
        n_mels = z.shape[0]
        plotter.scatter(
            func_name="Image", source=png_str, row=row,
            x0=float(self.times[0]),
            dx=float((self.times[-1] - self.times[0]) / (len(self.times) - 1)),
            y0=0,
        )
        plotter.update_layout(
            yaxis={
                "range": [0, n_mels - 1],
                "scaleanchor": False,
            }
        )
        return super().plot(show, row)


@dataclass(slots=True)
class DetectionResult:
    """ Hold Detection Result """
    sr: int
    extractor: Extractor
    audio: AudioNumpy
    
    rms: Rms
    flux: Flux
    mel: Mel

    _times: np.ndarray | None = field(init=False, default=...)
    _final_mask: np.ndarray | None = field(init=False, default=...)
    _segments: list[AudioSegment] | None = field(init=False, default=...)

    def mask_times(self, safe_start: float = 0.0) -> tuple[float, float]:
        scipy = self.extractor.scipy
        find_objects, label = scipy.ndimage.find_objects, scipy.ndimage.label
        shifted_times = self.times + safe_start
        slices = [slc[0] for slc in find_objects(label(self.final_mask)[0])]
        return [
            (float(round(shifted_times[slc.start], 2)), float(round(shifted_times[slc.stop-1], 2)))
            for slc in slices
        ]
    
    @property
    def final_mask(self) -> np.ndarray:
        if self._final_mask is not ...:
            return self._final_mask
        mask = self.rms.mask | self.mel.mask
        mask = self.extractor.mask_forward(mask, self.rms.valleys)
        mask = self.extractor.mask_merge(mask)
        mask = self.extractor.mask_delete(mask)
        mask = self.extractor.mask_merge_lower_duration(mask)
        self._final_mask = self._split_final_mask(mask)
        return self._final_mask

    def _split_final_mask(self, datamask: np.ndarray) -> np.ndarray:
        """ Split Long block Mask Width """
        maxframe = self.extractor.ms2frame(ExtractorConfig.max_maskblock_ms)
        minframe = self.extractor.ms2frame(ExtractorConfig.min_maskblock_ms)
        minsilenceframe = self.extractor.ms2frame(ExtractorConfig.min_silence_ms)
        starts, ends = self.extractor.get_mask_starts_ends(datamask)
        for start, end in zip(starts, ends):
            chunks = [(start, end)]
            while any((
                (ce-cs) >= maxframe
                for cs, ce in chunks
            )):
                new_chunks = []
                for cs, ce in chunks:
                    if (ce-cs) >= maxframe:
                        valleys = [
                            v for v in self.rms.valleys
                            if (cs+minframe) <= v <= (ce-minframe) # - gapframe?
                        ]
                        candidates = []
                        for v in valleys:
                            if not self.mel.mask[v]:
                                # mel mask is False, find star and end?
                                pastmask = self.mel.mask[cs:v]
                                aftermask = self.mel.mask[v:ce]
                                truemask = np.where(pastmask==True)[0]
                                if len(truemask) > 0: maskstart = cs + truemask[-1] + 1
                                else: maskstart = cs
                                truemask = np.where(aftermask==True)[0]
                                if len(truemask) > 0: maskend = v + truemask[0]
                                else: maskend = ce
                                assert maskend >= maskstart, (
                                    "Negative value for gap"
                                    f"\n  maskstart: {maskstart}"
                                    f"\n  maskend: {maskend}"
                                )
                                candidates.append({
                                    "start": maskstart,
                                    "end": maskend,
                                    "gap": maskend - maskstart
                                })
                        if candidates:
                            valid_candidates = [
                                c for c in candidates
                                if c["gap"] >= minsilenceframe
                            ]
                            if valid_candidates:
                                # pick the widest gap
                                best = max(valid_candidates, key=lambda c: c["gap"])
                                beststart, bestend = best["start"], best["end"]
                                datamask[beststart:bestend] = False
                                new_chunks.extend([(cs, beststart), (bestend, ce)])
                            else:
                                new_chunks.append((cs, ce))
                    else:
                        new_chunks.append((cs, ce))
                if chunks == new_chunks: break
                chunks = new_chunks
        return datamask
    
    @property
    def times(self) -> np.ndarray:
        if self._times is not ...:
            return self._times
        assert np.allclose(self.mel.times, self.rms.times) and np.allclose(self.flux.times, self.rms.times)
        self._times = self.rms.times
        return self._times

    @times.setter
    def times(self, value: np.ndarray) -> None:
        self._times = value

    @property
    def segments(self) -> list[AudioSegment]:
        if self._segments is not ...:
            return self._segments
        starts, ends = self.extractor.get_mask_starts_ends(self.final_mask)
        audiosegments = []
        for s, e in zip(starts, ends):
            audiosegments.append(AudioSegment(
                start=self.times[s],
                end=self.times[e] if e < len(self.times) else self.times[-1]
            ))
        self._segments = audiosegments
        return self._segments

    def plot(self) -> None:
        plotter = self.extractor.plotter
        self.mel.plot(row=1)
        self.rms.plot(row=2)
        self.flux.plot(row=3)
        plotter.scatter(
            row=4, x=self.times, y=self.final_mask * np.max(self.rms.smoothed),
            name="Final Mask",
            fill="tozeroy", line=dict(shape="hv"),
        )
        from kplus.tools.audio import Audio
        audio_uri = Audio.np2base64(self.audio, self.sr)
        plotter.update_titles(["Mel", "RMS", "Flux", "Final Mask"])
        plotter.show(audio_uri=audio_uri, segments=self.segments)


class AudioExtractor:
    """ Main Class for audio activity extractor """
    def __init__(self,
        precision_ms: float = 10,
        signal_overlap: float = 0.75,
        use_filter: bool = True,
        **kwargs
    ) -> None:
        self.precision_ms = precision_ms
        self.overlap = signal_overlap
        self.use_filter = use_filter
        self.extractor = Extractor(self.precision_ms, **kwargs)
        
    def frame_and_hop(self, sr: int) -> tuple[int, int]:
        hoplength = int((sr / 1000) * self.precision_ms) # if sr == 44100 and precision_ms == 1, hop_length = 44 samples
        framelength = round(hoplength / (1 - self.overlap)) # 150% of hop_length = 66 samples
        return hoplength, framelength

    def _preprocess_audio(self, audio: AudioType, sr: int) -> AudioNumpy:
        from kplus.tools.audio import Audio, AudioNumpy
        audionp = audio
        if not isinstance(audio, AudioNumpy):
            audionp = Audio(audio, samplerate=sr, channels=1).numpy
        assert isinstance(audionp, AudioNumpy)
        if self.use_filter:
            sos = self.extractor.scipy.signal.butter(10, [200, 5000], btype='bandpass', fs=sr, output='sos')
            audionp = self.extractor.scipy.signal.sosfilt(sos, audionp).astype(np.float32)
        return audionp

    def _detect_rms(self, audionp: AudioNumpy, sr: int) -> Rms:
        hoplength, framelength = self.frame_and_hop(sr)
        rms = self.extractor.librosa.feature.rms(y=audionp, frame_length=framelength, hop_length=hoplength)[0]
        return Rms(data=rms, sr=sr, extractor=self.extractor, hoplength=hoplength, framelength=framelength)
    
    def _detect_flux(self, audionp: AudioNumpy, sr: int) -> Flux:
        hoplength, framelength = self.frame_and_hop(sr)
        flux = self.extractor.librosa.onset.onset_strength(y=audionp, sr=sr, hop_length=hoplength, lag=1)
        return Flux(data=flux, sr=sr, extractor=self.extractor, hoplength=hoplength, framelength=framelength)

    def _detect_mel(self, audionp: AudioNumpy, sr: int) -> Mel:
        hoplength, framelength = self.frame_and_hop(sr)
        S = self.extractor.librosa.feature.melspectrogram(y=audionp, sr=sr, n_mels=256, n_fft=framelength, hop_length=hoplength)
        energy = S.sum(axis=0)
        return Mel(S=S, data=energy, sr=sr, extractor=self.extractor, hoplength=hoplength, framelength=framelength)

    def detect_rms(self, audio: AudioType, sr: int) -> Rms:
        audionp = self._preprocess_audio(audio, sr)
        rms = self._detect_rms(audionp, sr)
        if env.verbose: rms.plot(show=True)
        return rms
        
    def detect_flux(self, audio: AudioType, sr: int) -> Flux:
        audionp = self._preprocess_audio(audio, sr)
        flux = self._detect_flux(audionp, sr)
        if env.verbose: flux.plot(show=True)
        return flux

    def detect_mel(self, audio: AudioType, sr: int) -> Mel:
        audionp = self._preprocess_audio(audio, sr)
        mel = self._detect_mel(audionp, sr)
        if env.verbose: mel.plot(show=True)
        return mel
    
    def detect_all(self, audio: AudioType, sr: int, *, no_show: bool = False) -> DetectionResult:
        audionp = self._preprocess_audio(audio, sr)
        mel = self._detect_mel(audionp, sr)
        rms = self._detect_rms(audionp, sr)
        flux = self._detect_flux(audionp, sr)
        result = DetectionResult(
            sr=sr, extractor=self.extractor, audio=audionp,
            rms=rms, flux=flux, mel=mel,
        )
        if env.verbose and not no_show: result.plot()
        return result


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    path, sr = args[0], int(args[1])
    AudioExtractor().detect_all(path, sr)