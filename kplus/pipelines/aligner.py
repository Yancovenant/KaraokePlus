from __future__ import annotations
import string
import logging
import difflib
import re

from typing import TYPE_CHECKING, List
from types import SimpleNamespace
from itertools import pairwise

from kplus.environment import env
from .transcriber import WordTiming, Result, Segment
from .aad import AudioSegment

if TYPE_CHECKING:
    import torch
    import numpy as np
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
        line_to_seg: dict[int, list[AudioSegment]] = {}
        seg_index = {seg: i for i, seg in enumerate(audio_segments)}
        for idx in range(len(lines_list)):
            line_words = [w for w in lyrics_data if w.line_idx == idx]
            anchors = [w for w in line_words if w.start is not None]
            if anchors:
                all_line_segs = []
                for w in anchors:
                    mid = (w.start + w.end ) / 2
                    closest_seg = min(audio_segments,
                        key=lambda s: abs(mid - (s.start + s.end) / 2) if not (s.start<=mid<=s.end) else 0)                    
                    all_line_segs.append(closest_seg)
                # Make sure no segment left behind if its having no anchor in the middle
                observed = sorted(set(all_line_segs), key=lambda s: s.start)
                min_i = seg_index[observed[0]]
                max_i = seg_index[observed[-1]]
                # If the very first word of the line is missing its timestamp
                # Use the previous segment and merged it later
                if line_words[0].start is None:
                    prev = line_to_seg.get(idx - 1)
                    prev_max_i = seg_index[prev[-1]] if prev else -1
                    if min_i - 1 > prev_max_i:
                        min_i -= 1
                line_to_seg[idx] = audio_segments[min_i:max_i + 1]
            else:
                prev = line_to_seg.get(idx - 1)
                line_to_seg[idx] = [prev[-1] if prev else audio_segments[0]]
        # if segment is overlapping between lines. we wanna and merge the segment?
        # REMAKE THE AUDIO SEGMENT
        super_lines: list[tuple[list[WordTiming], AudioSegment]] = []
        sorted_indices = sorted(line_to_seg.keys())
        if sorted_indices:
            current_group_words = [w for w in lyrics_data if w.line_idx == sorted_indices[0]]
            current_group_segs = set(line_to_seg[sorted_indices[0]])
            for idx in sorted_indices[1:]:
                next_segs = set(line_to_seg[idx])
                next_words = [w for w in lyrics_data if w.line_idx == idx]
                if current_group_segs & next_segs:
                    # OVERLAP: Merge words and segments
                    logger.debug(f"overlap! {current_group_segs & next_segs}")
                    current_group_words.extend(next_words)
                    current_group_segs.update(next_segs)
                else:
                    logger.debug("Chain broken. Starting a new group...")
                    min_start = min(s.start for s in current_group_segs)
                    max_end = max(s.end for s in current_group_segs)
                    super_lines.append((current_group_words, AudioSegment(start=min_start, end=max_end)))

                    current_group_words = next_words
                    current_group_segs = next_segs
            # Finalize the very last group after loop ends
            if current_group_words:
                min_start = min(s.start for s in current_group_segs)
                max_end = max(s.end for s in current_group_segs)
                super_lines.append((current_group_words, AudioSegment(start=min_start, end=max_end)))
        new_transcribe_segments = []
        new_audio_segments = []
        for words, segment in super_lines:
            # Dropped word
            allowed_start = segment.start - max_bleed_sec
            allowed_end = segment.end + max_bleed_sec
            n = len(words)
            for i in range(n):
                if words[i].start is None:
                    logger.debug(f"{'':<2} Dropped: {words[i].word}")
                    prev_end = allowed_start
                    for left in range(i - 1, -1, -1):
                        if words[left].end is not None:
                            logger.debug(f"{'':<4} Found leading end: {words[left].end}")
                            prev_end = words[left].end
                            break
                    next_start = allowed_end
                    for right in range(i + 1, n):
                        if words[right].start is not None:
                            logger.debug(f"{'':<4} Found trailing start: {words[right].start}")
                            next_start = words[right].start
                            break
                    missing_block = []
                    for j in range(i, n):
                        if words[j].start is None:
                            missing_block.append(j)
                        else:
                            break
                    logger.debug(f"{'':<6} total dropped in between: {len(missing_block)}")
                    gap = next_start - prev_end
                    time_per_word = gap / (len(missing_block) + 1)
                    curr_time = prev_end
                    for idx in missing_block:
                        logger.debug(f"{'':<8} Interpolate: [None - None] to [{curr_time:.2f}s - {curr_time + time_per_word:.2f}s]")
                        words[idx].start = curr_time
                        words[idx].end = curr_time + time_per_word
                        curr_time += time_per_word
            dropped_words = [w for w in words if w.start is None or w.end is None]
            if dropped_words:
                logger.warning(f"!!! THERE IS A DROP WORD STILL, possibly end time is missing, {dropped_words}")
            valid_starts = [w.start for w in words if w.start is not None]
            valid_ends = [w.end for w in words if w.end is not None]
            min_time = f"{min(valid_starts):.2f}" if valid_starts else "None"
            max_time = f"{max(valid_ends):.2f}" if valid_ends else "None"
            text_content = " ".join(w.word for w in words)
            logger.debug(f"word     >> [{min_time} - {max_time}] {text_content}")
            logger.debug(f"duration >> [{segment.start:.2f} - {segment.end:.2f}]")
            logger.debug("-" * 40)
            new_transcribe_segments.append(Segment(words=words))
            new_audio_segments.append(segment)
        
        return Result(segments=new_transcribe_segments), new_audio_segments

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


    def refine_segments_with_peaks(self, results: Result, audio_np: np.ndarray, precision_ms: int = 1):
        env.matplotlib, env.librosa
        import matplotlib.pyplot as plt, librosa
        for i, seg in enumerate(results.segments):
            safe_start = seg.start
            safe_end = seg.end
            start = int(safe_start * self.sr)
            end = int(safe_end * self.sr)
            audio_chunk = audio[start:end]
            # Will frame length actually be bigger then segment duration?
            # as long as its longer than 1ms * 1.5 its correct right
            hop_length = int(sr / 1000) * precision_ms
            frame_length = int(hop_length * 1.5) # 150% 
            sos = scipy.signal.butter(10, [300, 3000], btype='bandpass', fs=sr, output='sos')
            audio = scipy.signal.sosfilt(sos, audio)
            rms = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)[0]
            # lower this more probably for more precision?
            frames_per_half_sec = max(1, int(0.5 / (precision_ms / 1000)))
            rms_smoothed = uniform_filter1d(rms, size=frames_per_half_sec)
            inverted_rms = -rms_smoothed
            valleys, _ = find_peaks(inverted_rms, prominence=0.01)
            peak_prob_sec = 0.9 # 10% quieter ? to handle more verbose
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            valley_times = librosa.frames_to_time(valleys, sr=sr, hop_length=hop_length)
            if True:
                fig, axes = plt.subplots(2, 1, figsize=(50, 10), sharex=True)
                librosa.display.waveshow(audio_chunk, sr=self.sr, ax=axes[0], color='darkgray', alpha=0.5)
                axes[0].set_title("Audio Waveform")
                axes[0].set_ylabel("Amplitude")
                axes[1].plot(rms_times, rms_smoothed, label="Smoothed RMS", color='blue', linewidth=1.5)
                axes[1].plot(valley_times, rms_smoothed[valleys], "mo", markersize=8, label="Detected Deepest Peak")
                axes[1].set_title("RMS")
                axes[1].set_ylabel("RMS Amplitude")
                axes[1].set_xlabel("Time (s)")
                axes[1].legend(loc="upper right")
                axes[1].label_outer()
                plt.tight_layout()
                plt.show()
                plt.savefig(f"{i}_refinement.png", bbox_inches='tight')
                plt.close()
        return results

    def main(self, audio: torch.Tensor, sr: float, lyrics: str, audio_segments: List[AudioSegment], transcriptions: Result):
        env.demucs
        from .utils import convert_audio
        audio = convert_audio(audio, sr, self.sr, channels=1)
        audio_np = audio.detach().cpu().numpy().squeeze().copy()
        # Needleman Wunchs
        results, new_audio_segments = self.get_lyrics_timestamp(transcriptions, lyrics, audio_segments, 0.0)
        # Forced Align (FA)
        results = self.ctc_align(audio.to(env.device), results, new_audio_segments)
        
        # This uses rms peaks converted and normalized with filter
        results = self.refine_segments_with_peaks(results, audio_np, 1)
        
        logger.debug(f">> Final Timestamp {len(results.segments)}")
        for res in results.segments:
            logger.debug(f"{'':<2}[{res.start:.2f}s - {res.end:.2f}s] {res.text}")
            for w in res.words:
                logger.debug(f"{'':<4}WordTiming: {w.start:.2f}s to {w.end:.2f}s {w.word}")
        del audio, #audio_np
        env.clean()
        return results.to_lyrics_segment()