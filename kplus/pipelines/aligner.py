from __future__ import annotations
import string
import logging
import difflib
import re

from typing import TYPE_CHECKING, List
from types import SimpleNamespace

from kplus.environment import env
from .transcriber import WordTiming, Result, Segment

if TYPE_CHECKING:
    import torch
    import numpy as np
    from .aad import AudioSegment
    from .transcriber import Result


logger = logging.getLogger(__name__)


class Aligner:
    def __init__(self):
        env.torchaudio
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        self.sr = bundle.sample_rate
        self.model = bundle.get_model(with_star=True).to(env.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

        self.RE_CHINESE = re.compile(r'[\u4e00-\u9fff]+')
        self.RE_JP = re.compile(r'[\u3040-\u30ff]+')
        self.RE_KR = re.compile(r'[\uac00-\ud7af]+')
        self.RE_LATIN = re.compile(r'[^a-z]')

    def get_lyrics_timestamp(self, transcripts: Result, lyrics: str, audio_segments: List[AudioSegment], max_bleed_sec: float = 0.5) -> Result:
        def seg_index_for_time(start, end):
            if start is None or end is None:
                return None

            best_i = 0
            best_overlap = -1.0
            best_dist = float("inf")

            for i, s in enumerate(audio_segments):
                overlap = min(end, s.end) - max(start, s.start)
                if overlap < 0:
                    overlap = 0.0

                if overlap > 0:
                    dist = 0.0
                elif end < s.start:
                    dist = s.start - end
                elif start > s.end:
                    dist = start - s.end
                else:
                    dist = 0.0

                if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                    best_overlap = overlap
                    best_dist = dist
                    best_i = i

            return best_i
        def interpolate_words(words, allowed_start, allowed_end):
            n = len(words)
            i = 0

            while i < n:
                if words[i].start is not None:
                    i += 1
                    continue

                prev_end = allowed_start
                for left in range(i - 1, -1, -1):
                    if words[left].end is not None:
                        prev_end = words[left].end
                        break

                next_start = allowed_end
                for right in range(i + 1, n):
                    if words[right].start is not None:
                        next_start = words[right].start
                        break

                missing = []
                j = i
                while j < n and words[j].start is None:
                    missing.append(j)
                    j += 1

                gap = max(0.0, next_start - prev_end)
                step = gap / (len(missing) + 1)

                curr = prev_end
                for k in missing:
                    words[k].start = curr
                    words[k].end = curr + step
                    curr += step

                i = j
        env.sequence_align
        from sequence_align.pairwise import needleman_wunsch_with_scores
        def clean_word(w): return w.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        lines_list = [l.strip() for l in lyrics.split("\n") if l.strip() and not l.startswith('[')]
        lyrics_data = list(
            (obj := WordTiming(word=w, start=None, end=None, score=None,),
                setattr(obj, "clean", clean_word(w)),
                setattr(obj, "line_idx", i),
                setattr(obj, "seg_idx", None))[0]
            for i, line in enumerate(lines_list)
            for w in line.split())
        transcript_data = list(
            (obj := WordTiming(word=w.word, start=w.start, end=w.end, score=w.score),
                setattr(obj, "clean", clean_word(w.word)),
                setattr(obj, "seg_idx", segs.seg_idx))[0]
            for segs in transcripts.segments
            for w in segs.words if w.score > 0.35)
        lyrics_data_clean = [w.clean for w in lyrics_data]
        transcript_data_clean = [w.clean for w in transcript_data]
        logger.debug(f">> NeedlemanWunsch: ASR ({len(transcript_data)}), LYRICS ({len(lyrics_data)})")
        def fuzzy_score(a, b):
            if a == b: return 2.0  # Perfect match
            if difflib.SequenceMatcher(None, a, b).ratio() > 0.6: return 1.0  # High similarity (slight mishearings)
            return -3.0
        a, b = needleman_wunsch_with_scores(
            lyrics_data_clean, transcript_data_clean,
            gap="-", score_fn=fuzzy_score, indel_score=-1.0)
        ai, bi = 0, 0
        for a1, b1, in zip(a, b):
            logger.debug(f"{'':<2}Seq: ({a1}) <-> ({b1})")
            if a1 == "-": # whisper hallucination/words
                bi += 1
                continue
            if b1 == "-": # Whisper dropping word, pass?
                ai += 1
                continue
            else:
                if fuzzy_score(a1, b1) > 0:
                    logger.debug(f"{'':<4}Matched: {lyrics_data[ai].word} to {transcript_data[bi].word}")
                    lyrics_data[ai].score = transcript_data[bi].score
                    lyrics_data[ai].start = transcript_data[bi].start
                    lyrics_data[ai].end = transcript_data[bi].end
                    lyrics_data[ai].seg_idx = transcript_data[bi].seg_idx
                else:
                    logger.debug(f"{'':<4}!!! Not Matched?: {lyrics_data[ai].word} to {transcript_data[bi].word}")
                bi += 1
                ai += 1
        # Lines To Segment Calc
        # REVIEW: This would be perfectly fine for segments that hold multiple or merged the lines
        # But not entirely sure if its a splitted lines between smaller segments.
        # ! REVIEW IN PROGRESS
        line_to_seg = {}
        for idx in range(len(lines_list)):
            line_words = [w for w in lyrics_data if w.line_idx == idx]
            anchors = [w for w in line_words if w.start is not None and w.seg_idx is not None]
            if anchors:
                # Anchor is words that already have a segment in it. but there is also a dropped word
                seg_idxs = []
                last_si = None
                for w in anchors:
                    si = w.seg_idx
                    if last_si is not None and si < last_si:
                        # If the previous segment index is not none
                        # and current segment index is less than last_si ? how can this be
                        # so in a word timing it have, 4,4,5,5,4 ??
                        si = last_si
                        w.seg_idx = si
                    seg_idxs.append(si)
                    last_si = si
                min_i = min(seg_idxs)
                max_i = max(seg_idxs)
                line_to_seg[idx] = list(range(min_i, max_i + 1))
            else:
                prev = line_to_seg.get(idx - 1)
                # Previous line segment last segment
                line_to_seg[idx] = [prev[-1]] if prev else [0]
        for idx in range(len(lines_list)):
            line_words = [w for w in lyrics_data if w.line_idx == idx]
            span = line_to_seg[idx]
            if len(span) == 1:
                for w in line_words:
                    w.seg_idx = span[0]
                continue
            for w in line_words:
                si = getattr(w, "seg_idx", None)
                if si is None:
                    continue
                if si < span[0]:
                    w.seg_idx = span[0]
                elif si > span[-1]:
                    w.seg_idx = span[-1]
            
            allowed_start = audio_segments[span[0]].start - max_bleed_sec
            allowed_end = audio_segments[span[-1]].end + max_bleed_sec
            interpolate_words(line_words, allowed_start, allowed_end)
            for w in line_words:
                if getattr(w, "seg_idx", None) is None:
                    si = seg_index_for_time(w.start, w.end)

                    if si is None or si < span[0]:
                        si = span[0]
                    elif si > span[-1]:
                        si = span[-1]

                    w.seg_idx = si
            last_si = span[0]
            for w in line_words:
                if getattr(w, "seg_idx", None) is None or w.seg_idx < last_si:
                    w.seg_idx = last_si
                else:
                    last_si = w.seg_idx
        
        seg_buckets = [[] for _ in audio_segments]
        for w in lyrics_data:
            si = getattr(w, "seg_idx", None)

            if si is None:
                si = 0
                w.seg_idx = 0

            si = max(0, min(si, len(audio_segments) - 1))
            w.seg_idx = si
            seg_buckets[si].append(w)
                
        final_segments = []
        for seg_i, seg in enumerate(audio_segments):
            bucket = seg_buckets[seg_i]
            if not bucket: continue
            allowed_start = seg.start - max_bleed_sec
            allowed_end = seg.end + max_bleed_sec
            interpolate_words(bucket, allowed_start, allowed_end)
            # Rubber banding
            bucket_start = bucket[0].start
            bucket_end = bucket[-1].end

            target_start = seg.start
            target_end = seg.end

            dur_orig = bucket_end - bucket_start
            dur_new = target_end - target_start

            if dur_orig <= 0 or dur_new <= 0:
                step = max(0.0, dur_new) / max(1, len(bucket))
                curr = target_start

                for w in bucket:
                    w.start = curr
                    w.end = curr + step
                    curr += step
            else:
                scale = dur_new / dur_orig

                for w in bucket:
                    old_start = w.start
                    old_end = w.end

                    w.start = target_start + (w.start - bucket_start) * scale
                    w.end = target_start + (w.end - bucket_start) * scale

                    w.start = max(target_start, min(w.start, target_end))
                    w.end = max(target_start, min(w.end, target_end))

                    logger.debug(
                        f"{'':<6} {w.word} shifted "
                        f"[{old_start:.2f}s - {old_end:.2f}s] --> [{w.start:.2f}s - {w.end:.2f}s]"
                    )
            
            final_segments.append(Segment(words=list(
                (obj := WordTiming(word=w.word, start=w.start, end=w.end, score=w.score if w.score is not None else 0.0),
                    setattr(obj, "line_idx", w.line_idx))[0]
                for w in bucket)))
        final_segments.sort(key=lambda x: x.words[0].start)
        logger.debug(f">> Lyric Timestamp {len(final_segments)}")
        for seg in final_segments:
            logger.debug(f"{'':<2}[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
        return Result(segments=final_segments)

    def tokenize_line(self, text) -> List[SimpleNamespace]:
        tokens = []
        chunks = re.split(r'([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+)', text)
        for chunk in chunks:
            if not chunk: continue
            if (self.RE_JP.search(chunk) or
                self.RE_CHINESE.search(chunk) or
                self.RE_KR.search(chunk)):
                tokens.extend(SimpleNamespace(
                    original=char, token=token
                ) for char in chunk if (token:=self.RE_LATIN.sub('', char.lower())))
            else:
                prev_token = None
                for w in chunk.split():
                    if w in ".,!?" and prev_token: prev_token.original += w; continue
                    if token := self.RE_LATIN.sub('', w.lower()):
                        tokens.append(prev_token:=SimpleNamespace(
                        original=w, token=token))
        return tokens

    def ctc_align(self, audio: torch.Tensor, results: Result, audio_segments: List[AudioSegment]) -> Result:
        env.torch
        import torch
        for audio_segment in audio_segments:
            for segment in results.segments:
                if not segment.text:
                    continue
                start_seg = segment.start
                end_seg = segment.end
                midpoint = (start_seg + end_seg) / 2
                if audio_segment.start <= midpoint <= audio_segment.end:
                    tokens = self.tokenize_line(segment.text)
                    transcript_tokens = ["*"]
                    for tok in tokens:
                        transcript_tokens.extend(list(tok.token))
                        transcript_tokens.append("*")
                    safe_start = min(start_seg, audio_segment.start)
                    safe_end = max(end_seg, audio_segment.end)
                    start_sample = int(safe_start * self.sr)
                    end_sample = int(safe_end * self.sr)
                    audio_slice = audio[:, start_sample:end_sample]

                    if audio_slice.size(1) == 0:
                        del audio_slice
                        logger.warning("!!! Cannot process 0 duration audio to ctc alignment")
                        continue
                    try:
                        with torch.inference_mode():
                            emission, _ = self.model(audio_slice.to(env.device))
                            token_spans = self.aligner(emission[0], self.tokenizer(transcript_tokens))
                    except Exception as err:
                        logger.error(f"!!! Error while doing ctc align: {err}", exc_info=True)

                    if False:
                        pass # print plot alignments maybe

                    char_spans = [span for token, span in zip(transcript_tokens, token_spans) if token != "*"]
                    ratio = audio_slice.size(1) / emission.size(1)
                    char_idx = 0
                    for i, tok in enumerate(tokens):
                        word_len = len(tok.token)
                        current_char_spans = char_spans[char_idx : char_idx + word_len]
                        char_idx += word_len
                        if not current_char_spans:
                            continue
                        first_char_span = current_char_spans[0]
                        last_char_span = current_char_spans[-1]
                        local_start = int(ratio * first_char_span[0].start) / self.sr
                        local_end = int(ratio * last_char_span[-1].end) / self.sr
                        segment.words[i].start = local_start + safe_start
                        segment.words[i].end = local_end + safe_start
                    del token_spans, emission, audio_slice
        return results

    def track_vocal_tail(self, audio: np.ndarray, sr: int, start_time: float, end_search: float, fmin=50, fmax=1000) -> float:
        """ Uses pYIN to track when vocal pitch dies within a defined search window.
            Returns the adjusted end timestamp.
        """
        env.librosa, env.numpy
        import librosa, numpy as np
        # Slice the gap zone
        start_sample = int(start_time * sr)
        end_sample = int(end_search * sr)
        gap_audio = audio[start_sample:end_sample]

        if len(gap_audio) < 2048: return start_time # Too short for pYIN

        # Track pitch and voicing probability
        f0, voiced_flag, voiced_probs = librosa.pyin(
            gap_audio, sr=sr, fmin=fmin, fmax=fmax,
            frame_length=2048, hop_length=512
        )

        # Find the last frame where the voice is actually present (voiced_flag is True)
        # We look for the last index where voicing is detected
        voiced_indices = np.where(voiced_flag)[0]

        if len(voiced_indices) > 0:
            # Get time of the last voiced frame
            last_voiced_frame = voiced_indices[-1]
            tail_offset = librosa.frames_to_time(last_voiced_frame, sr=sr, hop_length=512)
            return start_time + tail_offset

        return start_time

    def refine_segments_with_dsp(self, results: Result, audio_np: np.ndarray, sr: int):
        for seg in results.segments:
            for i in range(len(seg.words) - 1):
                curr_word = seg.words[i]
                next_word = seg.words[i+1]
                gap = next_word.start - curr_word.end

                # If gap > 100ms, trigger DSP fallback
                if 0.1 < gap < 1.5: # 1.5s cap to prevent tracking background noise
                    new_end = self.track_vocal_tail(audio_np, sr, curr_word.end, next_word.start)
                    curr_word.end = new_end
        return results

    def main(self, audio: torch.Tensor, sr: float, lyrics: str, audio_segments: List[AudioSegment], transcriptions: Result):
        env.demucs
        from .utils import convert_audio
        audio = convert_audio(audio, sr, self.sr, channels=1)
        audio_np = audio.detach().cpu().numpy().squeeze().copy()
        # Needleman Wunchs
        results = self.get_lyrics_timestamp(transcriptions, lyrics, audio_segments, 0.0)
        # Forced Align (FA)
        results = self.ctc_align(audio.to(env.device), results, audio_segments)
        
        # This uses pyin maybe find another method if not already best
        results = self.refine_segments_with_dsp(results, audio_np, self.sr)
        
        logger.debug(f">> Final Timestamp {len(results.segments)}")
        for res in results.segments:
            logger.debug(f"{'':<2}[{res.start:.2f}s - {res.end:.2f}s] {res.text}")
            for w in res.words:
                logger.debug(f"{'':<4}WordTiming: ({w.score:.2f}) {w.start:.2f}s to {w.end:.2f}s {w.word}")
        del audio, #audio_np
        env.clean()
        return results.to_lyrics_segment()