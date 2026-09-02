from __future__ import annotations

import typing as t
import copy

from kplus import env
from kplus.tools.audio import Audio
from kplus.pipelines import detect_audio_activity
from kplus.pipelines.audio.plotter import AudioPlotter
from kplus.pipelines.utils import TextTiming, ASRResult

import numpy as np

if t.TYPE_CHECKING:
    from kplus.tools.audio import AudioType
    from kplus.pipelines.audio.detection import Mel, DetectionResult
    from kplus.pipelines.utils import AudioSegment
    

class Refiner:
    def __init__(self, **options):
        self.sr = 16000 # maybe?
        if env.verbose:
            self.plotter = AudioPlotter(options.pop("resample", False), shared_xaxes=True, vertical_spacing=0.05)
            
    def _compute_optimal(text: TextTiming, *ai_text_list, mel: Mel, safe_start: float) -> TextTiming:
        def _update_best(anchor):
            nonlocal best_start, best_end, best_score
            best_start, best_end, best_score = anchor.start, anchor.end, anchor.score
        valley_times = mel.valley_times(safe_start)
        for i, (*word_list,) in enumerate(zip(*(ai_text.words for ai_text in ai_text_list))):
            whs, qwn, mms = word_list
            best_start, best_end, best_score = None, None, None
            if mms.score > 0.1: _update_best(mms)
            if (whs.score > 0.1
                and whs.start > mms.end
            ): _update_best(whs) # always override
            if best_score is not None:
                midpoint = (best_start + best_end) / 2
                if (future_valleys := valley_times[valley_times > midpoint + 0.040]).size > 0:
                    best_end = future_valleys[np.abs(future_valleys - best_end).argmin()]
                if (past_valleys := valley_times[valley_times < midpoint - 0.040]).size > 0:
                    best_start = past_valleys[np.abs(past_valleys - best_start).argmin()]
            w = text.words[i]
            w.start, w.end, w.score = best_start, best_end, best_score
        return text

    def _compute_unreliable(text: TextTiming, *ai_text_list, mel: Mel, safe_start: float) -> TextTiming:
        def _update_best(anchor):
            nonlocal best_start, best_end, best_score
            best_start, best_end, best_score = anchor.start, anchor.end, anchor.score
        valley_times = mel.valley_times(safe_start)
        for i, (*word_list,) in enumerate(zip(*(ai_text.words for ai_text in ai_text_list))):
            w = text.words[i]
            prev_w = text.words[i-1] if i > 0 else None
            if w.score is not None: continue
            whs, qwn, mms = word_list
            best_start, best_end, best_score = None, None, None
            if (0 < qwn.duration < 4 # between 0-4 duration
               and (abs(mms.start - qwn.start) <= 1
               or abs(mms.end - qwn.end) <= 1 )
            ): _update_best(qwn)
            if best_score is None and mms.score > 0.001: _update_best(mms)
            if best_score is not None:
                if prev_w is not None:
                    best_start = max(prev_w.end, best_start)
                if i < len(text.words) - 1:
                    future_start = min(w.start for ai_text in ai_text_list for w in ai_text.words[i+1:])
                    if future_start:
                        best_end = min(future_start, best_end)
                midpoint = (best_start + best_end) / 2
                if (future_valleys := valley_times[valley_times > midpoint + 0.040]).size > 0:
                    best_end = future_valleys[np.abs(future_valleys - best_end).argmin()]
                if (past_valleys := valley_times[valley_times < midpoint - 0.040]).size > 0:
                    best_start = past_valleys[np.abs(past_valleys - best_start).argmin()]
            w.start, w.end, w.score = best_start, best_end, best_score
        return text

    def _compute_last(text: TextTiming, *ai_text_list, mask_times: tuple) -> TextTiming:
        for i, (*word_list,) in enumerate(zip(*(ai_text.words for ai_text in ai_text_list))):
            whs, qwn, mms = word_list
            w = text.words[i]
            prev_w = text.words[i-1] if i > 0 else None
            if w.score is None:
                w.start, w.end, w.score = prev_w.end, w.start, -1.0
            if prev_w is not None:
                w.start = max(prev_w.end, w.start)
            for mask_start, mask_end in mask_times:
                w.start = max(mask_start, w.start)
                if not (mask_end < mms.start or mask_start > mms.end):
                    # if mms is also overlapping
                    w.end = min(mask_end, w.end)
                    break
                else:
                    continue
                
    def _refine(self, o_text: TextTiming, *ai_text_list, safe_start: float, audio_result: DetectionResult) -> TextTiming:
        refined_text = copy.deepcopy(o_text)
        for w in refined_text.words: w.start = None; w.end = None; w.score = None
        refined_text = self._compute_optimal(refined_text, *ai_text_list, mel=audio_result.mel, safe_start=safe_start)
        refined_text = self._compute_unreliable(refined_text, *ai_text_list, mel=audio_result.mel, safe_start=safe_start)
        refined_text = self._compute_last(refined_text, *ai_text_list, mask_times=audio_result.mask_times(safe_start))
        return refined_text
    
    def __call__(self, audio: AudioType, ori: ASRResult, *ai_res, audiosegments: list[AudioSegment]) -> ASRResult:
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        refined = []
        for i, (o_text, *ai_text_list, aseg) in enumerate(zip(ori.texts, *(ai.texts for ai in ai_res), audiosegments)):
            assert all(len(ai_text.words) == len(o_text.words) for ai_text in ai_text_list), (
                "Not all align words have an equal number of words!"
                f"\nWord counts: {[(i, len(s.words)) for i, s in enumerate([o_text, *ai_text_list])]}"
                "\n"
                f"{'\n'.join(f'[{i}] ' + ' '.join(w.word for w in s.words) for i, s in enumerate([o_text, *ai_text_list]))}"
            )
            if env.verbose: self.plotter.refresh()
            min_ai_start = min(min(w.start for ai_text in ai_text_list for w in ai_text.words), (w.start for w in o_text.words))
            max_ai_end = max(max(w.end for ai_text in ai_text_list for w in ai_text.words), (w.end for w in o_text.words))
            safe_start = max(0, max(min(min_ai_start, aseg.start), min_ai_start - 1.0) - 0.5)
            safe_end = min(len(audionp), min(max(max_ai_end, aseg.end), max_ai_end + 1.0) + 0.5)
            audio_chunk = Audio.slicenp(audionp, safe_start, safe_end, self.sr)
            assert audio_chunk.shape[0] > 0, f"Audio shouldnt be 0 duration, {safe_start}-{safe_end}"

            audio_result = detect_audio_activity(audio=audionp, sr=self.sr)
            refined_text = self._refine(o_text, *ai_text_list, safe_start=safe_start, audio_result=audio_result)
            refined.append(refined_text)

            if env.verbose:
                self.plot(audio_result, refined_text, ai_text_list)
        return ASRResult(texts=refined)


    def plot(self, audio_result: DetectionResult, refined_text: TextTiming, *ai_text_list) -> None:
        audio_result.plot()