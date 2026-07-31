from __future__ import annotations

from typing import TYPE_CHECKING

from kplus.environment import env

from .utils import AudioSegment, Result, Segment, _process_audio

if TYPE_CHECKING:
    from .utils import AudioType

class Refiner:
    def __init__(self, sr=16000, precision_ms=0.5, verbose=False):
        self.sr = sr
        self.hop_length = int((self.sr / 1000) * precision_ms)
        self.frame_length = int(self.hop_length * 2)

    def _get_med_segs(self, ref_seg, *align_segs) -> Segment:
        env.numpy; import copy, numpy as np # type: ignore  # noqa: B018, I001
        # median_words = []
        for *words, r_word in zip(*(seg.words for seg in align_segs), ref_seg.words):
            med_start = np.median([w.start for w in words])
            med_end = np.median([w.end for w in words])
            r_word.start = float(med_start)
            r_word.end = float(med_end)
        return ref_seg

    def refine_timestamp(self, audio: AudioType, sr: int, *align_results, ref_segments: Result, audio_segments: list[AudioSegment]) -> Result:
        audio_np = _process_audio(audio, sr, self.sr)
        # refined_segs = []
        for i, (*align_segs, ref_seg, audio_segment) in enumerate(zip(*(res.segments for res in align_results), ref_segments.segments, audio_segments)):
            # min_align_start = min(seg.start for seg in align_segs)
            # safe_start = min(min_align_start, audio_segment.start)
            # max_align_end = max(seg.end for seg in align_segs)
            # safe_end = max(max_align_end, audio_segment.end)
            med_seg = self._get_med_segs(ref_seg, *align_segs)
            # refined_segs.append(med_seg)
        return ref_segments

#### More on the development later