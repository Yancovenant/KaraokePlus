
from typing import TYPE_CHECKING

from kplus.environment import env

from .utils import AudioSegment, Result, Segment, WordTiming, sec2ass, _process_audio

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType

class Refiner:
    def __init__(self, sr=16000, precision_ms=0.5, verbose=False):
        self.sr = sr
        self.hop_length = int((self.sr / 1000) * precision_ms)
        self.frame_length = int(self.hop_length * 2)

    def _get_med_segs(self, *align_segs) -> Segment:
        env.numpy; import copy, numpy as np # type: ignore  # noqa: B018, I001
        median_words = []
        for words in zip(*(seg.words for seg in align_segs)):
            med_start = np.median([w.start for w in words])
            med_end = np.median([w.end for w in words])
            rw = copy.copy(words[0])
            rw.start = float(med_start)
            rw.end = float(med_end)
            median_words.append(rw)
        return Segment(words=median_words)

    def refine_timestamp(self, audio: AudioType, sr: int, *align_results, audio_segments: list[AudioSegment]) -> Result:
        audio_np = _process_audio(audio, sr, self.sr)
        refined_segs = []
        for i, (*align_segs, audio_segment) in enumerate(zip(*(res.segments for res in align_results), audio_segments)):
            # min_align_start = min(seg.start for seg in align_segs)
            # safe_start = min(min_align_start, audio_segment.start)
            # max_align_end = max(seg.end for seg in align_segs)
            # safe_end = max(max_align_end, audio_segment.end)
            med_seg = self._get_med_segs(*align_segs)
            refined_segs.append(med_seg)
        return Result(segments=refined_segs)

#### More on the development later