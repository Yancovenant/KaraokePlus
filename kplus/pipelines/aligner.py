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
        env.sequence_align
        from sequence_align.pairwise import needleman_wunsch_with_scores
        def clean_word(w): return w.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        lines_list = [l.strip() for l in lyrics.split("\n") if l.strip() and not l.startswith('[')]
        lyrics_data = list(
            (obj := WordTiming(word=w, start=None, end=None, score=None,),
                setattr(obj, "clean", clean_word(w)),
                setattr(obj, "line_idx", i))[0]
            for i, line in enumerate(lines_list)
            for w in line.split())
        transcript_data = list(
            (obj := WordTiming(word=w.word, start=w.start, end=w.end, score=w.score),
                setattr(obj, "clean", clean_word(w.word)))[0]
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
                else:
                    logger.debug(f"{'':<4}!!! Not Matched?: {lyrics_data[ai].word} to {transcript_data[bi].word}")
                bi += 1
                ai += 1
        # Lines To Segment Calc
        # REVIEW: This would be perfectly fine for segments that hold multiple or merged the lines
        # But not entirely sure if its a splitted lines between smaller segments.
        line_to_seg = {}
        for idx in range(len(lines_list)):
            line_words = [w for w in lyrics_data if w.line_idx == idx]
            anchors = [w for w in line_words if w.start is not None]
            if anchors:
                seg_votes = []
                for w in anchors:
                    mid = (w.start + w.end ) / 2
                    closest_seg = min(audio_segments,
                        key=lambda s: abs(mid - (s.start + s.end) / 2) if not (s.start<=mid<=s.end) else 0)
                    seg_votes.append(closest_seg)
                line_to_seg[idx] = max(set(seg_votes), key=seg_votes.count)
            else:
                line_to_seg[idx] = line_to_seg.get(idx - 1, audio_segments[0])
        final_segments = []
        for seg in audio_segments:
            bucket = [w for w in lyrics_data if line_to_seg[w.line_idx] == seg]
            if not bucket: continue
            # Dropped word
            allowed_start = seg.start - max_bleed_sec
            allowed_end = seg.end + max_bleed_sec
            n = len(bucket)
            for i in range(n):
                if bucket[i].start is None:
                    logger.debug(f"{'':<2} Dropped: {bucket[i].word}")
                    prev_end = allowed_start
                    for left in range(i - 1, -1, -1):
                        if bucket[left].end is not None:
                            logger.debug(f"{'':<4} Found leading end: {bucket[left].end}")
                            prev_end = bucket[left].end
                            break
                    next_start = allowed_end
                    for right in range(i + 1, n):
                        if bucket[right].start is not None:
                            logger.debug(f"{'':<4} Found trailing start: {bucket[right].start}")
                            next_start = bucket[right].start
                            break
                    missing_block = []
                    for j in range(i, n):
                        if bucket[j].start is None:
                            missing_block.append(j)
                        else:
                            break
                    logger.debug(f"{'':<6} total dropped in between: {len(missing_block)}")
                    gap = next_start - prev_end
                    time_per_word = gap / (len(missing_block) + 1)

                    curr_time = prev_end
                    for idx in missing_block:
                        logger.debug(f"{'':<8} Interpolate: [None - None] to [{curr_time:.2f}s - {curr_time + time_per_word:.2f}s]")
                        bucket[idx].start = curr_time
                        bucket[idx].end = curr_time + time_per_word
                        curr_time += time_per_word

            # Rubber banding
            bucket_start = bucket[0].start
            bucket_end = bucket[-1].end
            target_start, target_end = bucket_start, bucket_end
            needs_correction = False
            if bucket_start < allowed_start:
                target_start = allowed_start
                needs_correction = True
                logger.debug(f"{'':<4} Word Early: {bucket_start:.2f}s --> {target_start:.2f}s")
            if bucket_end > allowed_end:
                target_end = allowed_end
                needs_correction = True
                logger.debug(f"{'':<4} Word Bleed: {bucket_end:.2f}s --> {target_end:.2f}s")
            if needs_correction:
                dur_orig = bucket_end - bucket_start
                dur_new = target_end - target_start
                scale = dur_new / dur_orig if dur_orig > 0 else 1.0

                for w in bucket:
                    old_start = w.start
                    old_end = w.end
                    w.start = target_start + (w.start - bucket_start) * scale
                    w.end = target_start + (w.end - bucket_start) * scale
                    logger.debug(f"{'':<6} {w.word} shifted [{old_start:.2f}s - {old_end:.2f}] --> [{w.start:.2f} - {w.end:.2f}]")

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

    def main(self, audio: torch.Tensor, sr: float, lyrics: str, audio_segments: List[AudioSegment], transcriptions: Result):
        env.demucs
        from .utils import convert_audio
        audio = convert_audio(audio, sr, self.sr, channels=1)
        # audio_np = audio.detach().cpu().numpy().squeeze().copy()
        # Needleman Wunchs
        results = self.get_lyrics_timestamp(transcriptions, lyrics, audio_segments, 0.0)
        # Forced Align (FA)
        results = self.ctc_align(audio.to(env.device), results, audio_segments)
        
        # This uses pyin maybe find another method if not already best
        # refine_results = self.refine_segments_with_dsp(results, audio_np, self.sr)
        logger.debug(f">> Final Timestamp {len(results.segments)}")
        for res in results.segments:
            logger.debug(f"{'':<2}[{res.start:.2f}s - {res.end:.2f}s] {res.text}")
            for w in res.words:
                logger.debug(f"{'':<4}WordTiming: ({w.score:.2f}) {w.start:.2f}s to {w.end:.2f}s {w.word}")
        del audio, #audio_np
        env.clean()
        return results.to_lyrics_segment()