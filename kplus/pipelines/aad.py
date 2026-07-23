from __future__ import annotations

import logging
import os

from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from kplus.tools.progress import MainProgress
from kplus.environment import env

if TYPE_CHECKING:
    from .utils import AudioType


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioSegment:
    start: float
    end: float

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, AudioSegment):
            return False
        return self.start == other.start and self.end == other.end


class AAD:
    """ Audio Activity Detection via RMS/DB 
    """
    def __init__(self, visualize: bool = False):
        self.visual = visualize
        env.matplotlib
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (20, 12)
        plt.rcParams['font.size'] = 10
    
    def plotvisual(self, audio, sr, start_times, end_times, valley_times, rms_times, rms_smoothed, valleys, raw_valleys, raw_valleys_times, silence_threshold):
        env.matplotlib, env.librosa
        import matplotlib.pyplot as plt, librosa
        fig, axes = plt.subplots(2, 1, figsize=(50, 10), sharex=True)
        librosa.display.waveshow(audio, sr=sr, ax=axes[0], color='darkgray', alpha=0.5)
        axes[0].set_title("Audio Waveform")
        axes[0].set_ylabel("Amplitude")
        for start, end in zip(start_times, end_times):
            axes[0].axvspan(start, end, color='green', alpha=0.2) 
            axes[0].axvline(x=start, color='green', linestyle='-', linewidth=1.5, alpha=0.8)
            axes[0].axvline(x=end, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[1].plot(rms_times, rms_smoothed, label="Smoothed RMS", color='blue', linewidth=1.5)
        axes[1].axhline(y=silence_threshold, color='black', linestyle='--', label="Silence Threshold")
        axes[1].plot(valley_times, rms_smoothed[valleys], "mo", markersize=8, label="Detected Deepest Peak")
        axes[1].plot(raw_valleys_times, rms_smoothed[raw_valleys], "mo", markersize=4, label="All Valleys")
        for start, end in zip(start_times, end_times):
            axes[1].axvspan(start, end, color='green', alpha=0.2)
        axes[1].set_title("RMS")
        axes[1].set_ylabel("RMS Amplitude")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc="upper right")
        axes[1].label_outer()
        plt.tight_layout()
        plt.show()
        plt.savefig("valleycuts.png", bbox_inches='tight')
        plt.close()
        
    def get_audio_segments(self, audio: AudioType,
                sr: int, precision_ms: int = 1, silence_threshold: int = 0.01,
                min_segment_sec: float = 2.0, peak_prob_sec: float = 8.0,
                depth_ratio: float = 0.6) -> List[AudioSegment]:
        """ Detect audio activity RMS, Peak, Voice 300fq - 3000fq
            Args:
                audio: AudioType
                sr: Sample rate
                precision_ms: in milisecond >= 1ms
                silence_treshold: anything below this value means silence
                min_segment_sec: segment needs to be minimal this value if it detect
                    a drop in the middle.
                peak_prob_sec: the peak will be compared by average mean value
                    of peak_prob_sec.
                depth_ratio: control how deep a peak should be to survive the average
                    of peak_prob_sec. 0.6 means a peak must be at least 40% quieter
        """
        env.scipy, env.librosa
        import librosa, scipy, numpy as np, torch
        from scipy.signal import find_peaks
        from scipy.ndimage import uniform_filter1d, median_filter
        if isinstance(audio, torch.Tensor):
            from .utils import convert_audio
            audio = convert_audio(audio, sr, sr, 1)
            audio = audio.detach().cpu().numpy().squeeze()
            assert sr is not None, "Passing torch.Tensor must be accompanied by its sample rate"
        elif not isinstance(audio, np.ndarray):
            audio, lsr = librosa.load(audio, sr=None)
            sr = lsr
            logger.debug(f"Changing sr value from {sr} to {lsr}")
        # If time manually given
        with MainProgress(total=5, desc=f"Processing audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}") as main_bar:
            main_bar.pbar.set_description("Computing RMS")
            hop_length = int(sr / 1000) * precision_ms
            frame_length = int(hop_length * 1.5) # 150% 
            sos = scipy.signal.butter(10, [300, 3000], btype='bandpass', fs=sr, output='sos')
            audio = scipy.signal.sosfilt(sos, audio)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            main_bar.update(1)
            main_bar.pbar.set_description("Smoothing out RMS")
            frames_per_half_sec = max(1, int(0.5 / (precision_ms / 1000)))
            rms_smoothed = uniform_filter1d(rms, size=frames_per_half_sec)
            main_bar.update(1)
            
            inverted_rms = -rms_smoothed
            min_segment_frames = int(min_segment_sec / (precision_ms / 1000))
            main_bar.pbar.set_description("Finding out peak on an invertes RMS")
            raw_valleys, _ = find_peaks(inverted_rms, prominence=0.01)
            peak_prob_frames = max(1, int(peak_prob_sec / (precision_ms / 1000)))
            local_mean = uniform_filter1d(rms_smoothed, size=peak_prob_frames)
            valleys = []
            for v in raw_valleys:
                if rms_smoothed[v] < (local_mean[v] * depth_ratio):
                    valleys.append(v)
            valleys = np.array(valleys)
            main_bar.update(1)
            main_bar.pbar.set_description("Building up segments last")
            segments = []
            current_start = None
            for i in range(len(rms_smoothed)):
                # 1. If we hit a flat-line (silence), treat it as a silence breath gap
                if rms_smoothed[i] < silence_threshold:
                    if current_start is not None:
                        if ((end_frame := i - 1) - current_start) < min_segment_frames and segments:
                            # merge it if this chunk is too tiny
                            prev_start, _ = segments[-1]
                            segments[-1] = (prev_start, end_frame)
                        else:
                            segments.append((current_start, end_frame))
                        current_start = None
                    continue
                # 2. If we come out of a gap, start a new segment
                if current_start is None:
                    current_start = i
                    continue
                # 3. If we hit a peak, cut exactly at the lowest point!
                if i in valleys:
                    # Only cut if the segment is long enough
                    if (i - current_start) > min_segment_frames:
                        segments.append((current_start, i))
                        current_start = i  # Start next segment immediately (touching)
            # Catch the final segment at the end of the song
            if current_start is not None and current_start < len(rms_smoothed) - 1:
                end_frame = len(rms_smoothed) - 1
                if (end_frame - current_start) < min_segment_frames and segments:
                    prev_start, _ = segments[-1]
                    segments[-1] = (prev_start, end_frame)
                else:
                    segments.append((current_start, len(rms_smoothed) - 1))
            main_bar.update(1)
            main_bar.pbar.set_description("Unpacking segments...")
            start_frames = np.array([seg[0] for seg in segments])
            end_frames = np.array([seg[1] for seg in segments])
            start_times = librosa.frames_to_time(start_frames, sr=sr, hop_length=hop_length)
            end_times = librosa.frames_to_time(end_frames, sr=sr, hop_length=hop_length)
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            valley_times = librosa.frames_to_time(valleys, sr=sr, hop_length=hop_length)
            raw_valleys_times = librosa.frames_to_time(raw_valleys, sr=sr, hop_length=hop_length)
            if self.visual:
                self.plotvisual(audio, sr, start_times, end_times, valley_times, rms_times, rms_smoothed, valleys, raw_valleys, raw_valleys_times, silence_threshold)
            logger.debug(f">> Total Audio Segment: {len(start_times)}")
            results = []
            for i, (start_t, end_t) in enumerate(zip(start_times, end_times)):
                logger.debug(f"{'':<2}{i+1}/{len(start_times)}: {start_t:.2f}s - {end_t:.2f}")
                results.append(AudioSegment(
                start=start_t, end=end_t))
            main_bar.update(1)
            return results
            