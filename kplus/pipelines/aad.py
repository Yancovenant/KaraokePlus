from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .utils import AudioSegment, _process_audio

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType


logger = logging.getLogger(__name__)


class AAD:
    """ Audio Activity Detection via RMS/DB 
    """
    def __init__(self, visualize: bool = False):
        self.visual = visualize
        env.matplotlib  # noqa: B018
        import matplotlib.pyplot as plt # type: ignore  # noqa: I001
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (20, 12)
        plt.rcParams['font.size'] = 10

        import plotly.graph_objects as go  # type: ignore
        self.go = go
    
    def plotvisual(self, audio, sr, start_times, end_times, valley_times, rms_times, rms_smoothed, valleys, raw_valleys, raw_valleys_times, silence_threshold):
        env.matplotlib, env.librosa  # noqa: B018
        import matplotlib.pyplot as plt, librosa # type: ignore  # noqa: I001
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
                sr: int, precision_ms: float = 0.5, silence_threshold: int = 0.01,
                min_segment_sec: float = 2.0, peak_prob_sec: float = 8.0,
                depth_ratio: float = 0.6) -> tuple[np.ndarray, list[AudioSegment]]:
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
        ### New code:
        env.scipy, env.librosa # noqa: B018
        import librosa, scipy, numpy as np # type: ignore  # noqa: I001
        from plotly.subplots import make_subplots # type: ignore
        from scipy.ndimage import uniform_filter1d, median_filter
        from scipy.ndimage import label, find_objects
        if sr is None and isinstance(audio, (str, Path)):
            env.demucs; from demucs.audio import AudioFile # type: ignore  # noqa: B018, I001
            sr = AudioFile(str(audio)).samplerate()
            logger.debug(f"Successfully get sr {sr}")
        audio = _process_audio(audio, sr, sr)
        with MainProgress(total=5, desc=f"Processing audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}") as main_bar:
            hop_length = int((sr / 1000) * precision_ms) # if sr == 44100 and precision_ms == 1, hop_length = 44 samples
            frame_length = int(hop_length * 1.5) # 150% of hop_length = 66 samples
            # Add filter?
            sos = scipy.signal.butter(10, [300, 3000], btype='bandpass', fs=sr, output='sos')
            audio = scipy.signal.sosfilt(sos, audio)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms)),sr=sr, hop_length=hop_length)
            uniform_f_length_sec = 0.5 #500ms original
            frames_per_half_sec = max(1, int(uniform_f_length_sec / (precision_ms / 1000)))
            rms_smoothed = uniform_filter1d(rms, size=frames_per_half_sec)

            # Peaks
            inverted_rms = -rms_smoothed
            raw_valleys = scipy.signal.find_peaks(inverted_rms, prominence=0.01)[0]

            rms_noise_floor = np.percentile(rms_smoothed, 5)
            rms_threshold = rms_noise_floor + (np.std(rms_smoothed) * 0.2)

            flux = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
            flux_scaled = (flux / np.max(flux)) * np.max(rms_smoothed) if np.max(flux) > 0 else flux
            flux_noise_floor = np.percentile(flux_scaled, 5)
            flux_threshold = flux_noise_floor + (np.std(flux_scaled) * 0.2)

            mask = (rms_smoothed > rms_threshold)

            onsets = librosa.onset.onset_detect(onset_envelope=flux, sr=sr, hop_length=hop_length,
                                            backtrack=True, normalize=True, delta=0.04)
            mask_flux = (flux_scaled > flux_threshold)
            valid_flux = mask_flux & (rms_smoothed > rms_noise_floor)

            combined_mask = mask | valid_flux
            # combined_mask[onsets[onsets < len(combined_mask)]] = False

            merge_window_ms = 140 # 140ms
            frames_to_bridge = max(1, int(merge_window_ms / precision_ms))
            final_mask = combined_mask.copy()
            lookahead_frames = max(1, int(150 / precision_ms))
            for o in onsets[onsets < len(final_mask)]:
                # If onset is outside a block, but a block starts shortly after...
                if not final_mask[o] and np.any(final_mask[o : o + lookahead_frames]):
                    # Find exactly where the block starts and fill the gap backward!
                    block_start = o + np.argmax(final_mask[o : o + lookahead_frames])
                    final_mask[o : block_start] = True
            # Find every frame where the mask turns OFF (the end of a block)
            falling_edges = np.where((final_mask[:-1] == True) & (final_mask[1:] == False))[0]
            max_forward_stretch_ms = 200 # Don't stretch more than 200ms into silence
            max_forward_frames = int(max_forward_stretch_ms / precision_ms)
            for edge in falling_edges:
                # Find the next valley that happens after this block ends
                valleys_after = raw_valleys[raw_valleys > edge]
                if len(valleys_after) > 0:
                    next_valley = valleys_after[0]
                    # Stretch forward to the valley, but cap it at max_forward_frames
                    end_point = min(next_valley, edge + max_forward_frames)
                    final_mask[edge : end_point] = True
            final_mask = median_filter(final_mask, size=frames_to_bridge)
            for v in raw_valleys:
                if final_mask[v]:
                    is_flux_dropping = (flux_scaled[v] <= flux_scaled[v-1]) if v > 0 else True
                    is_flux_low = flux_scaled[v] < flux_threshold
                    if is_flux_dropping and is_flux_low: final_mask[v] = False
            #
            if self.visual:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, row_heights=[0.5, 0.5, 0.5],
                    subplot_titles=("Raw Waveform", "RMS", "Segmented"))
                time_axis = np.linspace(0, len(audio) / sr, len(audio))
                fig.add_trace(self.go.Scattergl(x=time_axis, y=audio, name="Waveform",
                                line={"color": "#00d2ff", "width": 1}), row=1, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times, y=rms_smoothed, name="RMS Smooth", fill="tozeroy",
                    line={"color": "#ffaa00", "width": 2}
                ), row=2, col=1)
                fig.add_trace(self.go.Scattergl(
                        x=rms_times[raw_valleys], y=rms_smoothed[raw_valleys],
                        name="Valleys/Peaks", mode="markers", marker={'color': "red", 'size': 8, 'symbol': "circle"}
                    ), row=2, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times[onsets], y=rms_smoothed[onsets],
                    name="Onsets", mode="markers", marker={'color': "green", 'size': 8, 'symbol': "triangle-up"}
                ), row=2, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times, y=final_mask * np.max(rms_smoothed),
                    name="Final Mask", line={"color": "purple", "width": 2, "shape": "hv"},
                ), row=3, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times, y=combined_mask * np.max(rms_smoothed),
                    name="Combined Mask", line={"color": "red", "width": 1, "shape": "hv"},
                ), row=3, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times, y=flux_scaled, name="Spectral Flux",
                    line={"color": "#00ffcc", "width": 2} # Cyan color to stand out against orange
                ), row=2, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times, y=mask * np.max(rms_smoothed),
                    name="Mask (Active Audio)",
                    line={"color": "rgba(255, 0, 170, 0.5)", "width": 1, "shape": "hv"}, # 'hv' draws sharp 90-degree square waves
                ), row=2, col=1)
                fig.add_trace(self.go.Scattergl(
                    x=rms_times, y=mask_flux * np.max(rms_smoothed),
                    name="Mask (Spectral Flux)",
                    line={"color": "rgba(0, 255, 170, 0.5)", "width": 1, "shape": "hv"}, # 'hv' draws sharp 90-degree square waves
                ), row=2, col=1)
                fig.add_hline(y=flux_threshold, line_dash="dot", line_color="cyan", row=2, col=1,
                              annotation_text="Flux Threshold", annotation_position="top right")
                fig.add_hline(y=rms_threshold, line_dash="dash", line_color="green", row=2, col=1,
                              annotation_text="RMS Threshold", annotation_position="top left")
                fig.add_hline(y=rms_noise_floor, line_dash="dot", line_color="blue", row=2, col=1,
                              annotation_text="Noise Floor", annotation_position="top left")
                fig.update_layout(template="plotly_dark", hovermode="x unified",
                    height=675, margin={"l": 20, "r": 20, "t": 40, "b": 20},showlegend=False
                )
                fig.show()

            audio_segments = []
            labeled_array, num_features = label(final_mask)
            for i in range(1, num_features + 1):
                segment_slice = find_objects(labeled_array == i)[0]
                start_frame = segment_slice[0].start
                end_frame = segment_slice[0].stop - 1  # stop is exclusive, so -1 gets the last active frame
                
                start_t = rms_times[start_frame]
                end_t = rms_times[end_frame]
                
                # Optional: Filter out segments that are too short based on your function argument
                if (end_t - start_t) >= min_segment_sec:
                    audio_segments.append(AudioSegment(start=start_t, end=end_t))

        return audio, audio_segments
        ###

        env.scipy, env.librosa  # noqa: B018
        import librosa, scipy, numpy as np # type: ignore # noqa: I001
        from scipy.signal import find_peaks # type: ignore
        from scipy.ndimage import uniform_filter1d # type: ignore
        # Quick Workaround to get sr
        if sr is None and isinstance(audio, (str, Path)):
            env.demucs; from demucs.audio import AudioFile # type: ignore  # noqa: B018, I001
            sr = AudioFile(str(audio)).samplerate()
            logger.debug(f"Successfully get sr {sr}")
        audio = _process_audio(audio, sr, sr)
        # If time manually given maybe?
        with MainProgress(total=5, desc=f"Processing audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}") as main_bar:
            main_bar.pbar.set_description("Computing RMS")
            hop_length = int(sr / 1000) * precision_ms
            frame_length = int(hop_length * 1.5) # 150% 
            sos = scipy.signal.butter(10, [300, 3000], btype='bandpass', fs=sr, output='sos')
            audio = scipy.signal.sosfilt(sos, audio)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            main_bar.update(1)
            main_bar.pbar.set_description("Smoothing out RMS")
            uniform_f_length_sec = 0.060 #500ms original
            frames_per_half_sec = max(1, int(uniform_f_length_sec / (precision_ms / 1000)))
            rms_smoothed = uniform_filter1d(rms, size=frames_per_half_sec)
            main_bar.update(1)

            ### New
            rms_noise_floor = np.percentile(rms_smoothed, 5)
            rms_threshold = rms_noise_floor + (np.std(rms_smoothed) * 0.2)
            ###
            
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
            return audio, results
            