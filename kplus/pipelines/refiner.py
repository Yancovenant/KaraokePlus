from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.pipelines.aad import AAD
from kplus.tools.plotter import PlotterSimple

from .utils import (
    AudioLoader,
    AudioSegment,
    Result,
    Segment,
    WordTiming,
    _process_audio,
)

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType
    
logger = logging.getLogger(__name__)

class TimestampRefiner:
    def __init__(self, **kwargs):
        env.rich;  # noqa: B018
        from rich.console import Console  # type: ignore
        self.console = Console()
        self.np = env.numpy
        self.scipy = env.scipy  # type: ignore
        self.verbose = kwargs.pop("verbose", False)
        self.sr = kwargs.pop("sr", 16000) # asr model
        self.precision_ms = kwargs.pop("precision_ms", 0.5)
        self.hop_length = int((self.sr / 1000) * self.precision_ms)
        overlap = kwargs.pop("overlap", 0.75)
        self.frame_length = round(self.hop_length / (1 - overlap)) # 4 = 75% overlap
        if self.verbose:
            from rich.panel import Panel  # type: ignore
            self.panel = Panel
            self.plotter = PlotterSimple(kwargs.pop("resample", False), shared_xaxes=True, vertical_spacing=0.05)
        self.aad_class = AAD(SimpleNamespace(verbose=False, precision_ms=self.precision_ms, sr=self.sr, overlap=overlap), resample=False, )
        if kwargs:
            logger.warning("Unused Kwargs %s", (kwargs,))

    def _prepare_refinement(self, align_segs: list[Segment], audio_np: np.ndarray, ori_seg: Segment, aseg: AudioSegment) -> tuple[float, np.ndarray]:
        min_start_align_segs = min(align_segs, key=lambda x: x.start).start
        max_end_align_segs = max(align_segs, key=lambda x: x.end).end
        min_start = min(min_start_align_segs, ori_seg.start, aseg.start)
        max_end = max(max_end_align_segs, ori_seg.end, aseg.end)
        audio_chunk = self.slice_audio(audio_np, min_start, max_end)
        assert audio_chunk.shape[0] > 0, f"Audio shouldnt be 0 duration, {min_start}-{max_end}"
        return min_start, audio_chunk

    def slice_audio(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        return audio[int(start*self.sr):int(end*self.sr)]

    def display_audio(self, audio_np, refined_segment) -> None:
        from IPython.display import Audio, display  # type: ignore
        for w in refined_segment.words:
            if w.start is None or w.end is None: continue
            self.console.print(f"[{w.h_start}-{w.h_end}] ({w.duration:.3f}) {w.word} | {str(w.score)}")
            achunk= self.slice_audio(audio_np, w.start, w.end)
            if achunk.shape[0] > 0: display(Audio(achunk, rate=self.sr))
            else: self.console.print("No Audio")
            del achunk

    def _pop_kw(self, kwargs, *keys) -> tuple[SimpleNamespace, dict]:
        d = SimpleNamespace()
        for key in keys:
            setattr(d, key, kwargs.pop(key, None))
        return d, kwargs

    def _draw_seg(self, seg: Segment, color: str, row: int, max_y = 1) -> None:
        color = self.plotter.hex2rgba(color, 0.3)
        xcords, ycords, labels, text_y, text_x, h_texts = [],[],[],[],[],[]
        for i, word in enumerate(seg.words):
            if word.score is None: continue
            xcords.extend([*[word.start]*2, *[word.end]*2, None])
            ycords.extend([0, max_y, max_y, 0, None])
            score_str = f"{word.score:.3f}" if word.score is not None else "N/A"
            labels.append(f"({i})<br>{word.word}<br>{score_str}")
            text_x.append((word.start + word.end)/2)
            text_y.append(max_y/2)
            h_str = f"Word: {word.word}<br>[{word.start:.3f}-{word.end:.3f}] ({word.duration:.3f})"
            h_texts.extend([*[h_str]*4, None])
        self.plotter.scatter(x=xcords, y=ycords, name="", text=h_texts, hovertemplate="%{text}<extra></extra>", mode="lines", fill="toself", fillcolor=color, row=row, line=dict(width=2, color=color.replace("0.3", "0.8")))
        center_hovers = [f"Word: {word.word}<br>[{word.start:.3f}-{word.end:.3f}] ({word.duration:.3f})" for word in seg.words if word.score is not None]
        self.plotter.scatter(x=text_x, y=text_y, name="", customdata=center_hovers, hovertemplate="%{customdata}<extra></extra>", text=labels, textposition="middle center", row=row, textfont=dict(color="white", size=12), marker=dict(color=color.replace("0.3", "1.0"), size=6, symbol="square", line=dict(width=0)), mode="text+markers",)

    def _draw_align_segs(self, align_segs: list[Segment], colors: list[str], row: int) -> None:
        total_segs = len(align_segs)
        for s_idx, seg in enumerate(align_segs):
            color = colors[s_idx % len(colors)]
            color = self.plotter.hex2rgba(color, 0.3)
            xcords, ycords, labels, text_y, text_x, h_texts = [],[],[],[],[],[]
            plot_idx = total_segs - 1 - s_idx
            for w in seg.words:
                xcords.extend([w.start, w.start, w.end, w.end, None])
                ycords.extend([plot_idx, plot_idx+1, plot_idx+1, plot_idx, None])
                score_str = f"{w.score:.3f}" if w.score is not None else "N/A"
                labels.append(f"{w.word}<br>{score_str}")
                text_x.append((w.start + w.end)/2)
                text_y.append((plot_idx + plot_idx+1)/2)
                h_str = f"Word: {w.word}<br>[{w.start:.3f}-{w.end:.3f}] ({w.duration:.3f})"
                h_texts.extend([*[h_str]*4, None])
            self.plotter.scatter(x=xcords, y=ycords, name="", text=h_texts, hovertemplate="%{text}<extra></extra>", mode="lines", fill="toself", fillcolor=color, row=row, line=dict(width=2, color=color.replace("0.3", "0.8")))
            center_hovers = [f"Word: {w.word}<br>[{w.start:.3f}-{w.end:.3f}] ({w.duration:.3f})" for w in seg.words]
            self.plotter.scatter(x=text_x, y=text_y, name="", customdata=center_hovers, hovertemplate="%{customdata}<extra></extra>", text=labels, textposition="middle center", row=row, textfont=dict(color="white", size=12), marker=dict(color=color.replace("0.3", "1.0"), size=6, symbol="square", line=dict(width=0)), mode="text+markers",)

    def draw_verbose(self, **kwargs) -> None:
        if self.verbose:
            librosa = env.librosa
            d, kwargs = self._pop_kw(kwargs, "audio_np",
                "segment", "audio_c", "min_start", "aad_res",
                "align_segs")
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
            self.plotter.update_titles(["Mel Spectogram", "Combine Flux, RMS, Energy, Onset Strength, Refined Segment", "3 AI Alignment"])
            if d.align_segs is not None:
                self._draw_align_segs(d.align_segs, colors, row=3)
            if d.aad_res is not None:
                shifted_times = d.aad_res.rms_times + d.min_start
                flux_norm = d.aad_res.flux_smoothed * self.np.max(d.aad_res.rms_smoothed)
                onset_times = shifted_times[d.aad_res.onsets]
                y_min, y_max = 0, self.np.max(d.aad_res.rms_smoothed)
                v_lines_x, v_lines_y = [], []
                for t in onset_times:
                    v_lines_x.extend([t, t, None])
                    v_lines_y.extend([y_min, y_max, None])
                self.plotter.scatter(row=2, x=shifted_times, y=d.aad_res.final_mask * self.np.max(d.aad_res.rms_smoothed), name="Final Mask",
                                     fill="tozeroy", line={"shape": "hv", "color": "rgba(0,255,170,0.1)", "width": 1})
                self.plotter.scatter(row=2, x=shifted_times, y=d.aad_res.rms_smoothed, name="RMS")
                self.plotter.scatter(row=2, x=shifted_times, y=flux_norm, name="Flux")
                self.plotter.scatter(row=2, x=v_lines_x, y=v_lines_y, name="Onset",
                                     line={"width": 1, "dash": "dash"})
                self.plotter.scatter(row=2, x=shifted_times[d.aad_res.onset2], y=flux_norm[d.aad_res.onset2], name="Onset2", mode="markers", marker=dict(symbol="circle", size=4, color="red"))
                self.plotter.scatter(row=2, x=shifted_times[d.aad_res.peak_onset], y=flux_norm[d.aad_res.peak_onset], name="Peak Onset", mode="markers", marker=dict(symbol="circle", size=8, color="blue"))
            if d.segment is not None:
                self._draw_seg(d.segment, row=2, color="AB63FA", max_y=self.np.max(d.aad_res.rms_smoothed))
            if d.audio_c is not None:
                shifted_times = d.aad_res.rms_times + d.min_start
                S = librosa.feature.melspectrogram(y=d.audio_c, sr=self.sr, n_mels=256, n_fft=self.frame_length, hop_length=self.hop_length)
                S_dB = librosa.power_to_db(S, ref=self.np.max)
                times = librosa.times_like(S, sr=self.sr, hop_length=self.hop_length) + d.min_start
                energy = S.sum(axis=0)
                energy = energy / self.np.max(energy)
                energy_norm = energy * self.np.max(d.aad_res.rms_smoothed)
                self.plotter.scatter(row=2, x=times, y=energy_norm, mode="lines", name="Total energy")
                self.plotter.scatter(row=1, func_name="Heatmap",
                    z=S_dB, x=times, colorscale="Magma", showscale=False)
                self.plotter.scatter(row=2, x=shifted_times[d.aad_res.energy_onset], y=energy_norm[d.aad_res.energy_onset], name="Energy Onset", mode="markers", marker=dict(symbol="circle", size=8, color="green"))
            self.plotter.show()
            if True and d.audio_np is not None and d.segment is not None:
                self.display_audio(d.audio_np, d.segment)

    def _refine_mask(self, aad_res) -> None:
        merge_gap_frames = max(1, int(80 / self.precision_ms)) # 80ms gap
        min_width_frames = max(1, int(40 / self.precision_ms)) # 40ms width
        aad_res.final_mask = self.aad_class._merge_gaps(aad_res.final_mask, merge_gap_frames)

    def _get_final_mask_times(self, aad_res, shifted_times):
        from scipy.ndimage import find_objects, label  # type: ignore
        labeled_mask, num_features = label(aad_res.final_mask)
        mask_slices = [slc[0] for slc in find_objects(labeled_mask)]
        mask_times = [(float(shifted_times[slc.start]), float(shifted_times[slc.stop])) for slc in mask_slices]
        return mask_times

    def _validate(self, segment: Segment) -> None:
        for i, w in enumerate(segment.words):
            assert w.score is not None, f"Score cannot be None, {w}"
            assert w.start is not None, f"Start cannot be None, {w}"
            assert w.end is not None, f"End cannot be None, {w}"
            next_w = segment.words[i+1] if i < len(segment.words) - 1 else None
            if next_w is not None:
                assert w.start < next_w.end, "Cannot orderly overlap"

    def _compute_optimal_pair(self, words: list[WordTiming], w: WordTiming, prev_w: WordTiming | None, next_words: list[WordTiming] | None,
                              onset_times2: np.ndarray, onset_times: np.ndarray, peak_onset_times: np.ndarray,
                              energy_onset_times: np.ndarray) -> None:
        best_start, best_end, best_score = None, None, None
        qwn, whs, mms = words[0], words[1], words[-1]
        prev_w = prev_w if prev_w is not None else SimpleNamespace(start=0.0)
        def _update_best(anchor):
            nonlocal best_start, best_end, best_score
            best_start, best_end, best_score = anchor.start, anchor.end, anchor.score
        if mms.score > 0.1: _update_best(mms)
        if whs.score > 0.1 and whs.start > mms.end: _update_best(whs)
        if best_start is not None and best_end is not None:
            mid = (best_start + best_end) / 2
            end_onset = self.np.concatenate((onset_times2[onset_times2 > mid], energy_onset_times[energy_onset_times > mid], onset_times[onset_times > mid]))
            if end_onset.size > 0:
                closest = self.np.abs(end_onset - best_end).argmin()
                best_end = end_onset[closest]
            safe_end = best_end - 0.040
            start_onset = self.np.concatenate((energy_onset_times[energy_onset_times < safe_end], onset_times2[onset_times2 < safe_end]))
            if start_onset.size > 0:
                closest = self.np.abs(start_onset - best_start).argmin()
                best_start = start_onset[closest]
        if best_start is None and best_end is None:
            print("None", w)
        w.start, w.end, w.score = best_start, best_end, best_score
    
    def _compute_low_conf_pair(self, words: list[WordTiming], w: WordTiming, prev_w: WordTiming | None, next_words: list[WordTiming] | None,
                              onset_times2: np.ndarray, onset_times: np.ndarray, peak_onset_times: np.ndarray,
                              energy_onset_times: np.ndarray) -> None:
        best_start, best_end, best_score = None, None, None
        qwn, whs, mms = words[0], words[1], words[-1]
        prev_w = prev_w if prev_w is not None else SimpleNamespace(start=0.0, end=0.0)
        def _update_best(anchor):
            nonlocal best_start, best_end, best_score
            best_start, best_end, best_score = anchor.start, anchor.end, anchor.score
        if (best_start is None 
            and best_end is None 
            and 0 < (qwn.end - qwn.start) < 4
            and mms.start - qwn.end <= 1
            and qwn.start - mms.end <= 1
        ): _update_best(qwn)
        if best_start is None and best_end is None and mms.score > 0.001: _update_best(mms)
        if best_start is not None and best_end is not None and prev_w.end is not None:
            best_start = max(prev_w.end, best_start)
        if best_start is not None and best_end is not None and next_words:
            valid_next_min_start = [t.start for t in next_words if t.start is not None]
            if len(valid_next_min_start) > 0:
                next_min_start = min(valid_next_min_start)
                best_end = min(next_min_start, best_end)
        if best_start is not None and best_end is not None:
            mid = (best_start + best_end) / 2
            end_onset = self.np.concatenate((onset_times2[onset_times2 > mid], energy_onset_times[energy_onset_times > mid], onset_times[onset_times > mid]))
            if end_onset.size > 0:
                closest = self.np.abs(end_onset - best_end).argmin()
                best_end = end_onset[closest]
            safe_end = best_end - 0.040
            start_onset = self.np.concatenate((energy_onset_times[energy_onset_times < safe_end], onset_times2[onset_times2 < safe_end]))
            if start_onset.size > 0:
                closest = self.np.abs(start_onset - best_start).argmin()
                best_start = start_onset[closest]
        w.start, w.end, w.score = best_start, best_end, best_score

    def _refine(self, align_segs: list[Segment], ori_seg: Segment, aad_res: SimpleNamespace, min_start: float) -> Segment:
        refined_segment = copy.deepcopy(ori_seg)
        for w in refined_segment.words: w.start = None; w.end = None; w.score = None

        mask_times = self._get_final_mask_times(aad_res, aad_res.rms_times + min_start)
        shifted_times = aad_res.rms_times + min_start
        onset_times = shifted_times[aad_res.onsets]
        onset_times2 = shifted_times[aad_res.onset2]
        peak_onset_times = shifted_times[aad_res.peak_onset]
        energy_onset_times = shifted_times[aad_res.energy_onset]

        for i, (*words,) in enumerate(zip(*(seg.words for seg in align_segs))):
            prev_w = refined_segment.words[i-1] if i > 0 else None
            next_words = [seg.words[i+1:] for seg in align_segs]
            w = refined_segment.words[i]
            self._compute_optimal_pair(words, w, prev_w, next_words, onset_times2, onset_times, peak_onset_times, energy_onset_times)
        for i, (*words,) in enumerate(zip(*(seg.words for seg in align_segs))):
            prev_w = refined_segment.words[i-1] if i > 0 else None
            next_words = refined_segment.words[i+1:]
            w = refined_segment.words[i]
            if w.score is None or w.start is None or w.end is None:
                self._compute_low_conf_pair(words, w, prev_w, next_words, onset_times2, onset_times, peak_onset_times, energy_onset_times)
        for i, (*words,) in enumerate(zip(*(seg.words for seg in align_segs))):
            prev_w = refined_segment.words[i-1] if i > 0 else None
            next_words = refined_segment.words[i+1:]
            w = refined_segment.words[i]
            if w.score is None or w.start is None or w.end is None:
                w.start = prev_w.end
                w.end = w.start
                w.score = -1
            if prev_w is not None:
                w.start = max(prev_w.end, w.start)
            for mask_start, mask_end in mask_times:
                if not (mask_end < w.start or mask_start > w.end):
                    w.start = max(mask_start, w.start)
                    qwn, whs, mms = words[0], words[1], words[-1]
                    if not (mask_end < mms.start or mask_start > mms.end):
                        # if mms is also overlapping
                        w.end = min(mask_end, w.end)
                        break
                    else:
                        continue
        self._validate(refined_segment)
        return refined_segment

    def refine(self, audio: AudioType, ori: Result, audio_segments: list[AudioSegment], *align_result) -> Result:
        librosa = env.librosa
        audio_loader = AudioLoader(audio, samplerate=self.sr, channels=1)
        audio_np = audio_loader.audio_np
        sos = self.scipy.signal.butter(10, [200, 5000], btype='bandpass', fs=self.sr, output='sos')
        audio_np = self.scipy.signal.sosfilt(sos, audio_np).astype(self.np.float32)
        refined_segments = []
        for i, (*align_segs, ori_seg, aseg) in enumerate(zip(*(res.segments for res in align_result), ori.segments, audio_segments)):
            assert all(len(seg.words) == len(ori_seg.words) for seg in align_segs), (
                "Not all align words have an equal number of words!"
                f"\nWord counts: {[(i, len(s.words)) for i, s in enumerate([ori_seg, *align_segs])]}"
                "\n"
                f"{'\n'.join(f'[{i}] ' + ' '.join(w.word for w in s.words) for i, s in enumerate([ori_seg, *align_segs]))}"
            )
            if self.verbose: self.plotter.refresh()
            min_start, audio_chunk = self._prepare_refinement(align_segs, audio_np, ori_seg, aseg)
            aad_res = self.aad_class.get_final_mask(audio_chunk)
            self._refine_mask(aad_res)
            flux_norm = aad_res.flux_smoothed * self.np.max(aad_res.rms_smoothed)
            aad_res.onset2 = librosa.onset.onset_detect(onset_envelope=flux_norm, sr=self.sr, hop_length=self.hop_length, backtrack=True)
            aad_res.peak_onset = librosa.onset.onset_detect(onset_envelope=flux_norm, sr=self.sr, hop_length=self.hop_length)

            S = librosa.feature.melspectrogram(y=audio_chunk, sr=self.sr, n_mels=256, n_fft=self.frame_length, hop_length=self.hop_length)
            energy = S.sum(axis=0)
            energy = energy / self.np.max(energy)
            energy_norm = energy * self.np.max(aad_res.rms_smoothed)
            aad_res.energy_onset = librosa.onset.onset_detect(onset_envelope=energy_norm, sr=self.sr, hop_length=self.hop_length, backtrack=True)
            
            refine_segment = self._refine(align_segs, ori_seg, aad_res, min_start)
            refined_segments.append(refine_segment)
            if self.verbose: self.draw_verbose(
                audio_np=audio_np, segment=refine_segment, audio_c=audio_chunk,
                min_start=min_start, aad_res=aad_res, align_segs=align_segs)

        return Result(segments=refined_segments)