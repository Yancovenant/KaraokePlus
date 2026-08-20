from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .utils import AudioLoader, AudioSegment, _process_audio

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType


logger = logging.getLogger(__name__)


class AAD:
    """ Audio Activity Detection via RMS/DB 
    """
    def __init__(self, options, resample=True):
        env.librosa, env.scipy # noqa: B018
        import scipy; self.scipy = scipy  # type: ignore  # noqa: I001
        import librosa; self.librosa = librosa  # type: ignore  # noqa: I001
        self.precision_ms = options.precision_ms
        self.verbose = options.verbose
        self.sr = None
        overlap = 0.75
        if (overlap:=options.overlap) is not None:
            assert 0 <= overlap < 1, "Overlap cannot be negative or more than 1"
        if options.sr is not None:
            self._populate_sr(options.sr, overlap)
        if self.verbose:
            env.plotly  # noqa: B018
            import plotly.graph_objects as go  # type: ignore
            from plotly.subplots import make_subplots  # type: ignore
            if resample:
                env.plotly_resampler  # noqa: B018
                from plotly_resampler import register_plotly_resampler  # type: ignore
                register_plotly_resampler(mode='auto')
            self.go = go
            self.make_subplots = make_subplots

    def _populate_sr(self, sr, overlap):
        self.sr = sr
        self.hop_length = int((self.sr / 1000) * self.precision_ms) # if sr == 44100 and precision_ms == 1, hop_length = 44 samples
        self.frame_length = int(round(self.hop_length / (1 - overlap))) # 150% of hop_length = 66 samples

    def _merge_gaps(self, mask: np.ndarray, merge_gap_frame: int) -> np.ndarray:
        """Fill False gaps shorter than `merge_gap_frame` that lie between two True blocks."""
        env.numpy; import numpy as np  # type: ignore  # noqa: B018, I001
        out = mask.copy()
        diffs = np.diff(np.concatenate(([0], out.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0] 
        for s, e in zip(ends[:-1], starts[1:]): # i.e. the end of one True block, the start of the next True block
            gap_length = e - s
            if gap_length <= merge_gap_frame:
                out[s:e] = True
        return out

    def _delete_mask(self, mask: np.ndarray, min_width_frame: int) -> np.ndarray:
        """Delete True blocks shorter than `min_width_frame`."""
        env.numpy; import numpy as np  # type: ignore  # noqa: B018, I001
        out = mask.copy()
        diffs = np.diff(np.concatenate(([0], out.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            mask_length = e - s
            if mask_length <= min_width_frame:
                out[s:e] = False
        return out

    def _get_rms(self, audio, row=2):
        env.numpy; import numpy as np  # type: ignore  # noqa: B018, I001
        from scipy.ndimage import uniform_filter1d  # type: ignore
        rms = self.librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        times = self.librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=self.hop_length)
        uniform_f_length_sec = 0.060 #500ms original
        frames_per_half_sec = max(1, int(uniform_f_length_sec / (self.precision_ms / 1000)))
        rms_smoothed = uniform_filter1d(rms, size=frames_per_half_sec)

        rms_noise_floor = np.percentile(rms_smoothed, 5)
        rms_threshold = rms_noise_floor + (np.std(rms_smoothed) * 0.2)
        rms_mask = (rms_smoothed > rms_threshold)

        if self.verbose:
            self.fig.add_trace(self.go.Scattergl(
                x=times, y=rms_mask * np.max(rms_smoothed),
                name="Mask (Active Audio)", fill="tozeroy",
                line={"color": "red", "width": 1, "shape": "hv"}, # 'hv' draws sharp 90-degree square waves
            ), row=row, col=1)
            self.fig.add_trace(self.go.Scatter(
                x=times, y=rms_smoothed, name="RMS Smooth", fill="tozeroy",
                    line={"color": "#ffaa00", "width": 2}
            ), row=row, col=1)
            self.fig.add_hline(y=rms_threshold, line_dash="dot", line_color="magenta", row=row, col=1,
                annotation_text="RMS Threshold", annotation_position="top right")
        return times, rms_smoothed, rms_mask

    def _get_valleys(self, rms_smoothed, rms_times, row=2):
        inverted_rms = -rms_smoothed
        raw_valleys = self.scipy.signal.find_peaks(inverted_rms, prominence=0.01)[0]
        if self.verbose:
            self.fig.add_trace(self.go.Scattergl(
                x=rms_times[raw_valleys], y=rms_smoothed[raw_valleys],
                name="Valleys/Peaks", mode="markers", marker={'color': "red", 'size': 8, 'symbol': "circle"}
            ), row=row, col=1)
        return raw_valleys

    def _get_flux(self, audio, rms_smoothed, rms_times, rms_mask, row=2):
        import numpy as np  # type: ignore
        from scipy.ndimage import uniform_filter1d  # type: ignore
        flux = self.librosa.onset.onset_strength(y=audio, sr=self.sr, hop_length=self.hop_length, lag=1)
        flux_scaled = (flux / np.max(flux)) #* np.max(rms_smoothed) if np.max(flux) > 0 else flux
        times = self.librosa.times_like(flux, sr=self.sr, hop_length=self.hop_length)

        uniform_f_length_sec = 0.060 #500ms original
        frames_per_half_sec = max(1, int(uniform_f_length_sec / (self.precision_ms / 1000)))
        flux_smoothed = uniform_filter1d(flux_scaled, size=frames_per_half_sec)

        flux_noise_floor = np.percentile(flux_smoothed, 5)
        flux_threshold = flux_noise_floor + (np.std(flux_smoothed) * 0.2)
        mask_flux = (flux_smoothed > flux_threshold) & rms_mask
        onsets = self.librosa.onset.onset_detect(
            onset_envelope=flux_smoothed, sr=self.sr, hop_length=self.hop_length,
            backtrack=True, delta=0.01, #normalize=False, delta=0.04
        )
        if self.verbose:
            self.fig.add_trace(self.go.Scattergl(
                x=times, y=mask_flux * np.max(flux_smoothed),
                name="Mask (Spectral Flux)", fill="tozeroy",
                line={"color": "cyan", "width": 1, "shape": "hv"}, # 'hv' draws sharp 90-degree square waves
            ), row=row, col=1)
            self.fig.add_trace(self.go.Scattergl(
                x=times, y=flux_smoothed, name="Flux Smoothed",
                line={"color":"#00ffcc", "width": 3},
            ), row=row, col=1)
            self.fig.add_trace(self.go.Scattergl(
                x=times[onsets], y=flux_smoothed[onsets],
                name="Onsets", mode="markers", marker={'color': "black", 'size': 4, 'symbol': "circle"}
            ), row=row, col=1)
            self.fig.add_hline(y=flux_threshold, line_dash="dot", line_color="cyan", row=2, col=1,
                annotation_text="Flux Threshold", annotation_position="top right")
        return mask_flux, onsets, flux_smoothed, flux

    def _get_final_mask(self, rms_mask, flux_mask, rms_times, rms_smoothed, onsets, raw_valleys, row=3):
        import numpy as np  # type: ignore
        merge_gap_frames = max(1, int(140 / self.precision_ms)) # 140ms gap
        min_width_frames = max(1, int(100 / self.precision_ms)) # 100ms width
        raw_mask = rms_mask | flux_mask

        lookahead_frames = max(1, int(150 / self.precision_ms))
        for o in onsets[onsets < len(raw_mask)]:
            if not raw_mask[o] and np.any(raw_mask[o : o + lookahead_frames]):
                block_start = o + np.argmax(raw_mask[o : o + lookahead_frames])
                raw_mask[o : block_start] = True
        
        falling_edges = np.where((raw_mask[:-1] == True) & (raw_mask[1:] == False))[0]
        max_forward_stretch_ms = 200 # Don't stretch more than 200ms into silence
        max_forward_frames = int(max_forward_stretch_ms / self.precision_ms)
        for edge in falling_edges:
            valleys_after = raw_valleys[raw_valleys > edge]
            if len(valleys_after) > 0:
                next_valley = valleys_after[0]
                end_point = min(next_valley, edge + max_forward_frames)
                raw_mask[edge : end_point] = True

        merged_mask = self._merge_gaps(raw_mask, merge_gap_frames)
        final_mask = self._delete_mask(merged_mask, min_width_frames)
        max_block_frames = int(10000 / self.precision_ms)
        gap_frames = max(1, int(0.1 / self.precision_ms))
        flux_tolerance_frames = int(50 / self.precision_ms)
        diffs = np.diff(np.concatenate(([0], final_mask.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        min_width_frames = max(1, int(1000 / self.precision_ms)) # 1000ms width
        for s, e in zip(starts, ends):
            chunks = [(s, e)]
            while any((ce - cs) > max_block_frames for cs, ce in chunks):
                new_chunks = []
                for cs, ce in chunks:
                    if (ce - cs) > max_block_frames:
                        # Find valleys safely inside the chunk (preserving 1000ms minimum width on both sides)
                        valid_valleys = [v for v in raw_valleys if cs + min_width_frames <= v <= ce - min_width_frames - gap_frames]
                        safe_split_points = []
                        for v in valid_valleys:
                            win_start = max(0, v - flux_tolerance_frames) # -50ms
                            win_end = min(len(flux_mask), v + flux_tolerance_frames) # +50ms
                            window_flux = flux_mask[win_start:win_end]
                            # If there is ANY False in this window, it's not solid flux
                            if not np.all(window_flux):
                                # Find exactly WHICH frames in this window have False flux
                                false_flux_local_indices = np.where(window_flux == False)[0]
                                false_flux_global_indices = false_flux_local_indices + win_start

                                # Out of those False flux frames, find the exact one with the lowest volume
                                best_false_idx = min(false_flux_global_indices, key=lambda idx: rms_smoothed[idx])
                                left = best_false_idx
                                while left > 0 and not flux_mask[left]: left -= 1
                                right = best_false_idx
                                while right < len(flux_mask) - 1 and not flux_mask[right]: right += 1
                                flux_gap_width = right - left
                                safe_split_points.append({
                                    'idx': best_false_idx,
                                    'gap_width': flux_gap_width
                                })

                        if safe_split_points:
                            # 1. Reject plosive (like 'd', 'p', 'b')
                            min_word_gap_frames = int(10 / self.precision_ms) # e.g., 10ms
                            valid_splits = [s for s in safe_split_points if s['gap_width'] >= min_word_gap_frames]

                            if valid_splits:
                                # 2. Pick the split with the WIDEST flux gap (the clearest breath)
                                best_split = max(valid_splits, key=lambda s: s['gap_width'])
                                final_split = best_split['idx']

                                final_mask[final_split : final_split + gap_frames] = False
                                new_chunks.extend([(cs, final_split), (final_split + gap_frames, ce)])
                            else:
                                new_chunks.append((cs, ce))
                    else:
                        new_chunks.append((cs, ce))

                if chunks == new_chunks: break # Failsafe: exit if no splits were made
                chunks = new_chunks
        if self.verbose:
            self.fig.add_trace(self.go.Scattergl(
                x=rms_times, y=final_mask * np.max(rms_smoothed),
                name="Mask (Combined)", fill="tozeroy",
                line={"color": "rgba(0, 255, 170, 0.5)", "width": 1, "shape": "hv"}, # 'hv' draws sharp 90-degree square waves
            ), row=row, col=1)
        return final_mask

    def get_audio_segments(self, audio: AudioType) -> tuple[np.ndarray, list[AudioSegment]]:
        env.numpy; import numpy as np # type: ignore  # noqa: B018, I001
        from scipy.ndimage import label, find_objects # type: ignore
        audio_loader = AudioLoader(audio, channels=1)
        if self.sr == None:
            self.sr = audio_loader.samplerate()
            self._populate_sr(self.sr)
        audio = audio_loader.audio_np
        sos = self.scipy.signal.butter(10, [200, 5000], btype='bandpass', fs=self.sr, output='sos')
        audio = self.scipy.signal.sosfilt(sos, audio).astype(np.float32)
        self.fig = None
        if self.verbose:
            self.fig = self.make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, row_heights=[0.5, 0.5, 0.5],
                    subplot_titles=("Raw Waveform", "RMS", "Segmented"))
        rms_times, rms_smoothed, rms_mask = self._get_rms(audio)
        raw_valleys = self._get_valleys(rms_smoothed, rms_times)
        flux_mask, onsets, flux_smoothed, flux = self._get_flux(audio, rms_smoothed, rms_times, rms_mask)
        final_mask = self._get_final_mask(rms_mask, flux_mask, rms_times, rms_smoothed, onsets, raw_valleys)
        if self.verbose:
            self.fig.update_layout(template="plotly_dark", hovermode="x unified",
                height=675, margin={"l": 20, "r": 20, "t": 40, "b": 20},showlegend=False
            )
            self.fig.show()
        audio_segments = []
        labeled_array, num_features = label(final_mask)
        for i in range(1, num_features + 1):
            segment_slice = find_objects(labeled_array == i)[0]
            start_frame = segment_slice[0].start
            end_frame = segment_slice[0].stop - 1  # stop is exclusive, so -1 gets the last active frame
            start_t = rms_times[start_frame]
            end_t = rms_times[end_frame]
            audio_segments.append(AudioSegment(start=start_t, end=end_t))
        return audio, audio_segments

    def get_final_mask(self, audio_np):
        if self.verbose:
            self.fig = self.make_subplots(rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.05, subplot_titles=("RMS", "Flux", "Final"))
        rms_times, rms_smoothed, rms_mask = self._get_rms(audio_np, row=1)
        raw_valleys = self._get_valleys(rms_smoothed, rms_times, row=1)
        flux_mask, onsets, flux_smoothed, flux = self._get_flux(audio_np, rms_smoothed, rms_times, rms_mask, row=2)
        final_mask = self._get_final_mask(rms_mask, flux_mask, rms_times, rms_smoothed, onsets, raw_valleys, row=3)
        if self.verbose:
            self.fig.update_layout(template="plotly_dark", hovermode="x unified",
                height=675, margin={"l": 20, "r": 20, "t": 40, "b": 20},showlegend=False
            )
            self.fig.show()
        return SimpleNamespace(
            final_mask=final_mask, flux_mask=flux_mask, onsets=onsets, raw_valleys=raw_valleys,
            rms_times=rms_times, rms_smoothed=rms_smoothed, rms_mask=rms_mask, flux_smoothed=flux_smoothed,
            flux=flux,
        )