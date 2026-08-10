from __future__ import annotations

import difflib
import logging
import re
import string
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .aad import AudioSegment
from .transcriber import Result, Segment, WordTiming
from .utils import _process_audio

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType

env.sequence_align, env.pypinyin, env.pykakasi, env.anyascii, env.jellyfish  # noqa: B018
# Need to be below this line
from kplus.tools.romaji_converter import RomajiPhonetic  # noqa: I001
from sequence_align.pairwise import needleman_wunsch_with_scores  # type: ignore

logger = logging.getLogger(__name__)

class AlignerAny:

    def __init__(self):
        self.is_qwen = False
        self.is_whisper = False

    def populate_model(self, model, **kwargs):
        self.sr = 16000 # qwen and whisper
        self.model = model
        self.is_whisper = model.__module__.startswith('faster_whisper.')
        self.is_qwen = model.__module__.startswith('qwen_asr.')
        self.token_step = kwargs.pop("token_step", 0)
        self.regroup = kwargs.pop("regroup", False)
        return kwargs

    def get_default_model(self):
        env.torchaudio; import torchaudio  # type: ignore # noqa: B018, I001
        bundle = torchaudio.pipelines.MMS_FA
        self.sr = bundle.sample_rate
        self.model = bundle.get_model(with_star=True).to(env.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.RE_CHINESE = re.compile(r'[\u4e00-\u9fff]+')
        self.RE_JP = re.compile(r'[\u3040-\u30ff]+')
        self.RE_KR = re.compile(r'[\uac00-\ud7af]+')
        self.RE_LATIN = re.compile(r'[^a-z]')

    def default_tokenize(self, text) -> list[SimpleNamespace]:
        tokens = []
        for chunk in text.split():
            if not chunk.strip(): continue
            romaji = RomajiPhonetic(chunk)
            tokens.append(SimpleNamespace(original=romaji.orig, token=romaji.latin))
        return tokens

    def default_align(self, audio_slice: np.ndarray, res: Segment) -> SimpleNamespace:
        env.torch; import torch  # type: ignore # noqa: B018, I001
        audio_slice = torch.from_numpy(audio_slice).unsqueeze(0)
        tokens = self.default_tokenize(res.text)
        transcript_tokens = ["*"]
        for tok in tokens:
            transcript_tokens.extend(list(tok.token))
            transcript_tokens.append("*")
        try:
            with torch.inference_mode():
                emission, _ = self.model(audio_slice.to(env.device))
                token_spans = self.aligner(emission[0], self.tokenizer(transcript_tokens))
        except Exception as err:
            logger.error(f"!!! Error while doing ctc align: {err}", exc_info=True)
        char_spans = [span for token, span in zip(transcript_tokens, token_spans) if token != "*"]
        ratio = audio_slice.size(1) / emission.size(1)
        del token_spans, emission, audio_slice
        return SimpleNamespace(
            tokens=tokens, char_spans=char_spans, ratio=ratio
        )

    def _fix_whisper(self, new_res, res: Segment) -> None:
        ori_word = [RomajiPhonetic(w.word).latin for w in res.words]
        new_word = [RomajiPhonetic(w.word.strip()).latin for w in new_res.words]
        healed_words = []
        matcher = difflib.SequenceMatcher(None, ori_word, new_word) # should this converted to a number for faster performance? like jiwer does it
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            print(tag, i1, i2)
            if tag == 'equal' or tag == 'replace':
                healed_words.extend(new_res.words[j1:j2])
            elif tag == 'delete':
                logger.warning(f"  -> Whisper deleting words {i1} - {i2}")
                for missing_idx in range(i1, i2):
                    healed_words.append(res.words[missing_idx]) 
            elif tag == 'insert':
                logger.warning(f"  -> Dropping Whisper hallucination: {[w.word for w in new_res.words[j1:j2]]}")
        new_res.words = healed_words
        return new_res

    def align(self,
              model,
              audio: AudioType,
              sr: int,
              result: Result,
              audio_segments: list[AudioSegment],
              **kwargs) -> Result:
        if model is None:
            import copy
            self.get_default_model()
            is_default_model = True
        else:
            options = self.populate_model(model, **kwargs)
            is_default_model = False
        audio = _process_audio(audio, sr, self.sr)
        results = []
        desc = "Qwen" if self.is_qwen else ("Whisper" if self.is_whisper else "MMS_FA") + " Aligning..."
        with MainProgress(total = len(audio_segments), desc=desc, unit="chunk") as main_bar:
            try:
                prev_lang = None
                if self.is_qwen:
                    audio_chunk_list, text_chunk_list, lang_chunk_list, saved_safe_start = [], [], [], []
                for res, seg in zip(result.segments, audio_segments):
                    if seg.start <= ((res.end + res.start) / 2) <= seg.end:
                        safe_start = max(min(res.start, seg.start), res.start - 1.0)
                        safe_end = min(max(res.end, seg.end), res.end + 1.0)
                        start_sample = int(safe_start * self.sr)
                        end_sample = int(safe_end * self.sr)
                        audio_slice = audio[start_sample:end_sample]
                        assert audio_slice.shape[0] > 0
                        if isinstance(res.language, list):
                            if "en" in res.language:
                                language = "en"
                            else:
                                language = res.language[0] # first index?
                        else:
                            language = res.language
                        if language is None: language = prev_lang if prev_lang is not None else "en"
                        prev_lang = language
                        if self.is_qwen:
                            saved_safe_start.append(safe_start)
                            audio_chunk_list.append((audio_slice, self.sr))
                            text_chunk_list.append(res.latin)
                            lang_chunk_list.append(language) # Fallback currently to "en"
                        elif self.is_whisper:
                            align_results = self.model.align(
                                audio_slice, res.latin, token_step=self.token_step, regroup=self.regroup,
                                language=language, **options) # Auto lang later on.
                            align_results = self.model.refine(
                                audio_slice, align_results, steps="se",
                                precision=0.02, **options)
                            seg_words = []
                            for new_res in align_results.segments:
                                # Fix whisper hallucination
                                if len(new_res.words) != len(res.words):
                                    logger.warning(f"Words length missmatch for whisper force align {len(new_res.words)} -> {len(res.words)}")
                                    new_res = self._fix_whisper(new_res, res)
                                assert len(new_res.words) == len(res.words), f"Word missmatch {new_res.text}"
                                for word in new_res.words:
                                    seg_words.append(WordTiming(
                                        start=float(safe_start + word.start),
                                        end=float(safe_start + word.end),
                                        score=float(word.probability),
                                        word=str(word.word.strip())
                                    ))
                            results.append(Segment(words=seg_words))
                            main_bar.update(1)
                        elif is_default_model:
                            align_results = self.default_align(audio_slice, res)
                            char_idx, char_spans, ratio = 0, align_results.char_spans, align_results.ratio
                            seg_words = []
                            for i, tok in enumerate(align_results.tokens):
                                word_len = len(tok.token)
                                current_char_spans = char_spans[char_idx : char_idx + word_len]
                                char_idx += word_len
                                if not current_char_spans: continue
                                first_char_span = current_char_spans[0]
                                last_char_span = current_char_spans[-1]
                                local_start = int(ratio * first_char_span[0].start) / self.sr
                                local_end = int(ratio * last_char_span[-1].end) / self.sr
                                total_score = sum(c[0].score for c in current_char_spans) / len(current_char_spans)
                                rw = copy.copy(res.words[i])
                                rw.start = local_start + safe_start
                                rw.end = local_end + safe_start
                                rw.score = float(total_score)
                                seg_words.append(rw)
                            results.append(Segment(words=seg_words))
                            main_bar.update(1)
                if self.is_qwen:
                    align_results = self.model.forced_aligner.align(
                        audio=audio_chunk_list, text=text_chunk_list, language=lang_chunk_list,
                    )
                    for res, safe_start in zip(align_results, saved_safe_start):
                        seg_words = []
                        for w in res:
                            seg_words.append(WordTiming(
                                start=float(safe_start + w.start_time),
                                end=float(safe_start + w.end_time),
                                score=1.0,
                                word=str(w.text)
                            ))
                        results.append(Segment(words=seg_words))
                        main_bar.update(1)
            except:
                logger.exception("Error while doing alignment")
                raise
            finally:
                if is_default_model:
                    del self.model, self.tokenizer, self.aligner
                else:
                    del self.model
                env.clean()
                results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        return Result(segments=results)

    def get_reference_timestamp(self, hypothesis: Result, reference: str, audio_segments: list[AudioSegment]) -> tuple[Result, list[AudioSegment]]:
        return ReferenceAligner(verbose=self.verbose).get_reference_timestamp(hypothesis, reference, audio_segments)

###
# TImestamp aligner
###
class BasicList(list):
    pass

def score_fn(a, b):
    if a == b: return 2.0 # Match exactly
    aphone = ReferenceAligner.get_phonetic(a)
    bphone = ReferenceAligner.get_phonetic(b)
    ratio = difflib.SequenceMatcher(None, aphone.latin, bphone.latin).ratio()
    if ratio >= 0.6: return 1.0
    if aphone == bphone: return 1.0
    return -3.0 # Mismatched

class ReferenceAligner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            from rich.console import Console  # type: ignore
            self.console = Console()

    def prepare_alignments(self, hypothesis: Result, reference: str,) -> tuple[list[str], list[WordTiming], list[WordTiming]]:
        ref_lines = [line.strip() for line in reference.split("\n") if line.strip() and not line.startswith('[')]
        ref_tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            (token:=WordTiming(word=word, start=None, end=None, score=None),
                setattr(token, "line_idx", line_idx),
                setattr(token, "language", None))[0]
            for line_idx, line in enumerate(ref_lines)
            for word in line.split()
        ]
        hyp_tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            (token:=WordTiming(word=word.word, start=word.start, end=word.end, score=word.score),
                setattr(token, "language", segment.language))[0]
            for segment in hypothesis.segments
            for word in segment.words if word.score > 0.1
        ]
        return ref_lines, ref_tokens, hyp_tokens

    _PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)

    def normalize(self, s: str) -> str:
        s = s.translate(self._PUNCTUATION_TRANSLATOR).lower().strip()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s) # Kaldi
        return s

    @classmethod
    @lru_cache(maxsize=2048)
    def get_phonetic(cls, word: str):
        return RomajiPhonetic(word)

    def _sequence_align(self, ref_tokens: list[WordTiming], hyp_tokens: list[WordTiming]) -> list:
        ref_clean = [self.normalize(t.word) for t in ref_tokens]
        hyp_clean = [self.normalize(t.word) for t in hyp_tokens]
        _ref, _hyp = needleman_wunsch_with_scores(ref_clean, hyp_clean, gap="-", score_fn=score_fn, indel_score=-1.0)
        stats = {"M": 0, "S": 0, "D": 0, "I": 0}
        aligned_map = []
        r_idx = h_idx = 0
        for r_tok, h_tok in zip(_ref, _hyp):
            op = "I" if r_tok == "-" else ("D" if h_tok == "-" else ("M" if r_tok == h_tok else "S"))
            stats[op] += 1
            aligned_map.append(SimpleNamespace(
                ref_idx=None if op == "I" else r_idx,
                hyp_idx=None if op == "D" else h_idx,
                op=op
            ))
            r_idx += (op != "I")
            h_idx += (op != "D")

        N = stats["M"] + stats["S"] + stats["D"] or 1
        errors = stats["S"] + stats["D"] + stats["I"]
        if errors/N > 0.5:
            logger.warning(f"Lyric Alignment may be inaccurate due to error rate of more than 50%: wer: {errors/N:.1%}")
        if self.verbose:
            from rich.panel import Panel  # type: ignore
            C = {"M": "green", "S": "yellow", "D": "red", "I": "cyan"}
            aligned = [(f"[{C[m.op]}]{r.ljust(max(len(r), len(h)))}[/]",
                        f"[{C[m.op]}]{h.ljust(max(len(r), len(h)))}[/]",
                        f"[{C[m.op]}]{m.op.ljust(max(len(r), len(h)))}[/]")
                        for r, h, m in zip(_ref, _hyp, aligned_map)]
            texts = []
            for i in range(0, len(aligned), 10):
                r_c, h_c, o_c = zip(*aligned[i:i+10])
                texts.append(f"[white bold]REF |[/] {' | '.join(r_c)}\n"
                             f"[white bold]HYP |[/] {' | '.join(h_c)}\n"
                             f"[white bold]OP  |[/] {' | '.join(o_c)}")
            subtitle = (f"[bold]WER: {errors/N:.1%}[/] | "
                        f"[green]M: {stats['M']/N:.1%}[/] | "
                        f"[yellow]S: {stats['S']/N:.1%}[/] | "
                        f"[red]D: {stats['D']/N:.1%}[/] | "
                        f"[cyan]I: {stats['I']/N:.1%}[/]")
            self.console.print(Panel("\n\n".join(texts), title="[bold]Alignment Needleman[/]", subtitle=subtitle, expand=False))
        return aligned_map

    def _prepare_audiosegment_mask(self, audio_segments: list[AudioSegment], fpms: float = 1.0) -> np.ndarray:
        env.numpy; import numpy as np # type: ignore  # noqa: B018, I001
        max_dur = audio_segments[-1].end
        total_frames = int(round(max_dur * fpms, 3) * 1000) + 1
        audio_mask = np.full(total_frames, -1, dtype=np.int32) # int
        for i, seg in enumerate(audio_segments):
            start = int(round(seg.start * fpms, 3) * 1000)
            end = int(round(seg.end * fpms, 3) * 1000)
            start_frame = max(0, min(start, total_frames))
            end_frame = max(0, min(end, total_frames))
            assert end_frame > start_frame, f"Invalid audio segment {i}: {seg.start} -> {seg.end}"
            audio_mask[start_frame:end_frame] = i
        return audio_mask

    def _get_audiosegment_ids(self, audio_mask_ms: np.ndarray, line_words: list[WordTiming], fpms: float = 1.0):
        env.numpy; import numpy as np # type: ignore  # noqa: B018, I001
        anchors = [w for w in line_words if w.start is not None]
        matched_segs = []
        if anchors:
            for w in anchors:
                start = int(round(w.start * fpms, 3) * 1000)
                end = int(round(w.end * fpms, 3) * 1000)
                start_frame = max(0, min(start, audio_mask_ms.size - 1))
                end_frame = max(0, min(end, audio_mask_ms.size))
                word_mask = audio_mask_ms[start_frame:end_frame]
                audio_seg_id = word_mask[word_mask >= 0]
                audio_seg_id = np.unique(audio_seg_id).tolist()
                matched_segs.extend(audio_seg_id)
                w.audio_seg_id = audio_seg_id
        return matched_segs

    def _interpolate_segments_lines(self, lines_map, audio_mask_ms, audio_segments):
        assert sorted(lines_map.keys()) == list(range(len(lines_map)))
        new_lines_map: dict[int, BasicList[WordTiming]] = {} # {line_idx: List[WordTiming]}
        for i, line_words in lines_map.items():
            matched_segs = self._get_audiosegment_ids(audio_mask_ms, line_words)
            line_words.audio_segment_ids = sorted(set(matched_segs))
            new_lines_map[i] = line_words
        assert sorted(new_lines_map.keys()) == list(range(len(new_lines_map)))
        for i, line_words in new_lines_map.items():
            if not line_words.audio_segment_ids:
                prev_aseg, next_aseg = None, None
                prev_gap, next_gap = 0,0
                for j in range(i - 1, -1, -1):
                    if (prev_aseg:=new_lines_map[j].audio_segment_ids):
                        break
                    prev_gap += 1
                for j in range(i + 1, len(new_lines_map)):
                    if (next_aseg:=new_lines_map[j].audio_segment_ids):
                        break
                    next_gap += 1
                if not prev_aseg: prev_aseg = [0]
                if not next_aseg: next_aseg = [len(audio_segments) - 1]
                start_idx = i - prev_gap
                end_idx = i + next_gap
                prev_audio_idx, next_audio_idx = max(prev_aseg), min(next_aseg)
                assert prev_audio_idx <= next_audio_idx, (f"Invalid dropped segment range: {prev_audio_idx} -> {next_audio_idx}")
                dropped_segments = [new_lines_map[k] for k in range(start_idx, end_idx + 1)]
                interpolate_audio_segment_ids = [k for k in range(prev_audio_idx, next_audio_idx + 1)]
                for drop in dropped_segments: drop.audio_segment_ids = interpolate_audio_segment_ids
        return new_lines_map

    def _update_ts_from_interpolation(self, line_words, step, total, anchor_point, direction = 1):
        taken = 0
        if direction > 0:
            for i in range(0, total):
                line_words[i].end = anchor_point + taken + step
                line_words[i].start = line_words[i].end - step
                taken += step
        else:
            for i in range(len(line_words)-total, len(line_words)):
                line_words[i].end = anchor_point + taken + step
                line_words[i].start = line_words[i].end - step
                taken += step

    def _interpolate_on_based_anchors(self, line_words, anchor_words, anchor, anchor_segment_id, anchor_point, segment_pos_ts, total_words, audio_segments, is_first_idx):
        if anchor_words is None:
            # either first or last line idx
            if anchor and anchor_point:
                step = abs(anchor_point - segment_pos_ts) / max(1, total_words)
                self._update_ts_from_interpolation(line_words, step, total_words, min(anchor_point, segment_pos_ts), direction = 1 if is_first_idx else -1)
            else: # everything dropped
                if len(line_words.audio_segment_ids) == len(audio_segments):
                    # transcript error, fully drop
                    for w in line_words:
                        w.start = audio_segments[0].start
                        w.end = audio_segments[-1].end
                else:
                    # since `line_words.audio_segment_ids` already interpolated to the next
                    # anchored segment, if the `line_words` is fully drop
                    # no matter the gap, we cannot know of this?
                    # interpolate this into the end maybe
                    if is_first_idx:
                        step = (audio_segments[max(line_words.audio_segment_ids)].end - audio_segments[0].start) / len(line_words)
                        self._update_ts_from_interpolation(line_words, step, len(line_words), audio_segments[0].start)
                    else:
                        step = (audio_segments[-1].end - audio_segments[min(line_words.audio_segment_ids)].start) / len(line_words)
                        self._update_ts_from_interpolation(line_words, step, len(line_words), audio_segments[min(line_words.audio_segment_ids)].start)
        else:
            anchor_words_segment_id = max(anchor_words.audio_segment_ids) # prev
            if not is_first_idx:
                anchor_words_segment_id = min(anchor_words.audio_segment_ids) # next
            anchor_words_segment = audio_segments[anchor_words_segment_id]
            if anchor_segment_id == anchor_words_segment_id:
                if anchor and anchor_point:
                    # this means, in this lines there is anchor in this line words
                    # also anchor seg == this seg, i guess we can safely interpolate this
                    step = abs(anchor_point - segment_pos_ts) / max(1, total_words)
                    self._update_ts_from_interpolation(line_words, step, total_words, min(anchor_point, segment_pos_ts), direction = 1 if is_first_idx else -1)
                else:
                    # Every of this line words is dropped.
                    anch_start = anchor_words_segment.start if is_first_idx else audio_segments[max(line_words.audio_segment_ids)].start
                    anch_end = audio_segments[max(line_words.audio_segment_ids)].end if is_first_idx else anchor_words_segment.end
                    step = (anch_end - anch_start) / len(line_words)
                    self._update_ts_from_interpolation(line_words, step, len(line_words), anch_start)
            else:
                # if current segment and anchor segment is different
                # means the current segment should be > prev segment? or < next segment
                if anchor and anchor_point:
                    gap_to_anchor_segment = abs(segment_pos_ts - (anchor_words_segment.end if is_first_idx else anchor_words_segment.start))
                    if gap_to_anchor_segment >= 2.0: # 2s
                        # if gap of audio is more than 2s, it doesnt make sense for this line words interpolated to the previous or next segment
                        # as our audio segment already mostly leaving only silences.
                        step = abs(anchor_point - segment_pos_ts) / max(1, total_words)
                        self._update_ts_from_interpolation(line_words, step, total_words, min(anchor_point, segment_pos_ts), direction = 1 if is_first_idx else -1)
                    else:
                        # This would be edge cases where the line words is either using
                        # previous, both or current segment
                        anchor_words_segment_point = anchor_words_segment.start if is_first_idx else anchor_words_segment.end
                        step = abs(anchor_point - anchor_words_segment_point) / max(1, total_words)
                        self._update_ts_from_interpolation(line_words, step, total_words, min(anchor_point, anchor_words_segment_point), direction = 1 if is_first_idx else -1)
                        if self.verbose: self.console.print(f">> using segment anchor, gap less than 2s, anchor={anchor_point}, anchor_seg={anchor_words_segment_point}, total={total_words}, step={step}, start={min(anchor_point, anchor_words_segment_point)}, isfirst={is_first_idx}, ")
                        min_audio_seg_id = min(min(line_words.audio_segment_ids), anchor_words_segment_id)
                        max_audio_seg_id = max(anchor_words_segment_id, max(line_words.audio_segment_ids))
                        line_words.audio_segment_ids = list(range(min_audio_seg_id, max_audio_seg_id + 1))
                else:
                    # everything dropped again?
                    # because if a `line_words` fully dropped, its already interpolated to the previous and next overlapping anchor segment.
                    raise ValueError("Not sure why even this step get triggered, need more data and edge cases")


    def _interpolate_word_first(self, lines_map, line_words, anchor, line_idx, gap_from_start, current_segment_start_id, audio_segments):
        # first word is dropped
        # Resolution:
        # Check next words, keep the gap
        # Check unused audio mask
        # Check gap to previous line_words end time, previous line_words unused audio segment end time
        current_segment_start = audio_segments[current_segment_start_id]
        current_anchor_word_start = min(anchor, key=lambda x: x.start).start if anchor else None
        prev_line_words = lines_map.get(line_idx - 1) if line_idx > 0 else None
        if prev_line_words is not None:
            prev_segment_id = max(prev_line_words.audio_segment_ids)
            assert current_segment_start_id >= prev_segment_id, f"not sure why ({line_idx}) word segment id is less than the next idx: prev:{prev_segment_id}, cur={current_segment_start_id}"
        self._interpolate_on_based_anchors(line_words, prev_line_words, anchor, current_segment_start_id, current_anchor_word_start, current_segment_start.start, gap_from_start, audio_segments, is_first_idx=True)

    def _interpolate_word_last(self, lines_map, line_words, anchor, line_idx, gap_from_end, current_segment_end_id, audio_segments):
        # last word is dropped
        # Resolution:
        # Check previous words, keep the gap
        # Check unused audio mask
        # Check gap to next line_words, next line_words unused start time
        current_segment_end = audio_segments[current_segment_end_id]
        current_anchor_word_end = max(anchor, key=lambda x: x.end).end if anchor else None
        next_line_words = lines_map.get(line_idx + 1) if line_idx < len(lines_map) - 1 else None
        if next_line_words is not None:
            next_segment_id = min(next_line_words.audio_segment_ids)
            assert current_segment_end_id <= next_segment_id, f"not sure why ({line_idx}) word segment id is more than the next idx: next={next_segment_id}, cur={current_segment_end_id}"
        self._interpolate_on_based_anchors(line_words, next_line_words, anchor, current_segment_end_id, current_anchor_word_end, current_segment_end.end, gap_from_end, audio_segments, is_first_idx=False)

    def _interpolate_word_middle(self, line_words):
        for i, w in enumerate(line_words):
            if w.start is None:
                assert i != 0, "Word first should be interpolated previously"
                assert i < len(line_words) - 1, "Word last should be interpolated previously"
                prev_gap, next_gap = 0, 0
                prev_word, next_word = None, None
                for j in range(i - 1, -1, -1):
                    if (prev_word:=line_words[j]).start is not None:
                        break
                    prev_gap += 1
                for j in range(i + 1, len(line_words)):
                    if (next_word:=line_words[j]).start is not None:
                        break
                    next_gap += 1
                assert (prev_word.start is not None) == (next_word.start is not None), "Dropped word already is interpolated, not sure why this doesnt have any anchor word"
                start_idx = i - prev_gap
                end_idx = i + next_gap
                dropped_words = [line_words[k] for k in range(start_idx, end_idx + 1)]
                step = (next_word.start - prev_word.end) / max(1, len(dropped_words))
                self._update_ts_from_interpolation(dropped_words, step, len(dropped_words), prev_word.end)
        min_audio_seg_id = min(line_words.audio_segment_ids)
        max_audio_seg_id = max(line_words.audio_segment_ids)
        line_words.audio_segment_ids = list(range(min_audio_seg_id, max_audio_seg_id + 1))

    def _interpolate_words_lines(self, line_idx, line_words, lines_map, audio_segments):
        anchor = []
        gap_from_start = float("inf")
        gap_from_end = -1
        # 0 means first and last is not None
        # >1 means the next step
        for i, w in enumerate(line_words):
            if w.start is not None:
                anchor.append(w)
                gap_from_end = len(line_words) - 1 - i
                gap_from_start = min(i, gap_from_start)
        word_first = line_words[0]
        word_last = line_words[-1]
        current_segment_start_id = min(line_words.audio_segment_ids)
        current_segment_end_id = max(line_words.audio_segment_ids)
        if word_first.start is None:
            self._interpolate_word_first(lines_map, line_words, anchor, line_idx, gap_from_start, current_segment_start_id, audio_segments)
        if word_last.start is None:
            self._interpolate_word_last(lines_map, line_words, anchor, line_idx, gap_from_end, current_segment_end_id, audio_segments)
        if any(w.start is None for w in line_words):
            self._interpolate_word_middle(line_words)

    def _normalize_audio_segments(self, audio_segments):
        for aseg in audio_segments:
            aseg.start = float(round(aseg.start, 3))
            aseg.end = float(round(aseg.end, 3))
        
    def _map_to_audio_segment(self, ref_tokens: list[WordTiming], audio_segments: list[AudioSegment]):
        assert audio_segments, "audio_segments is required"
        self._normalize_audio_segments(audio_segments)
        audio_segments = sorted(audio_segments, key=lambda x: x.start)
        audio_mask_ms = self._prepare_audiosegment_mask(audio_segments, 1.0)
        lines_map: dict[int, BasicList[WordTiming]] = defaultdict(BasicList) # {line_idx: List[WordTiming]}
        for token in ref_tokens: lines_map[token.line_idx].append(token)
        lines_map = self._interpolate_segments_lines(lines_map, audio_mask_ms, audio_segments)
        for idx, line_words in lines_map.items():
            any_drop = any(w.start is None for w in line_words)
            if any_drop:
                self._interpolate_words_lines(idx, line_words, lines_map, audio_segments)
        return lines_map

    def _validate_line_to_audio(self, line_words, audio_seg_ids, audio_mask, audio_segments):
        env.numpy; import numpy as np # type: ignore  # noqa: B018, I001
        matched_segs = self._get_audiosegment_ids(audio_mask, line_words)
        assert set(matched_segs) == audio_seg_ids, (
            f"Missmatch between physical timestamps and tracked segment IDs!\n"
            f"Physical Mask Result: {set(matched_segs)}\n"
            f"Tracked in Memory:    {audio_seg_ids}\n"
            f"Difference:           {set(matched_segs).symmetric_difference(audio_seg_ids)}\n\n"
            f"Line Text: '{' '.join(w.word for w in line_words)}'\n\n"
            f"Word Timestamps:\n" + 
            "\n".join(f" - {w.word:<10} : {w.h_start} ➔ {w.h_end} | LineIdx: {w.line_idx} | Score: {w.score}" for w in line_words) + "\n\n"
            f"Involved Audio Segments:\n" + 
            "\n".join(f" - Seg {idx:<2} : {audio_segments[idx].h_start} ➔ {audio_segments[idx].h_end}" 
                      for idx in sorted(set(matched_segs) | audio_seg_ids))
        )
        min_start = min(line_words, key=lambda x: x.start).start
        max_end = max(line_words, key=lambda x: x.end).end
        min_audio_start = audio_segments[min(audio_seg_ids)].start
        max_audio_end = audio_segments[max(audio_seg_ids)].end
        assert (np.isclose(min_start, min_audio_start, atol=5e-3) or min_start >= min_audio_start), f"Word timing is too early, word={min_start} to audio={min_audio_start}"
        assert (np.isclose(max_end, max_audio_end, atol=5e-3) or max_end <= max_audio_end), f"Word timing is too long, word={max_end} to audio={max_audio_end}"
        assert any(w.language is not None for w in line_words), f"WordLines require minimum 1 language, {line_words}"
        for w in line_words:
            assert w.line_idx is not None, f"Word object changed as it doesnt have `line_idx`: {w}"
            assert np.isfinite(w.start), w
            assert np.isfinite(w.end), w


    def _cluster_lines(self, lines_map, audio_segments):
        lines_map = dict(sorted(lines_map.items()))
        clusters = []
        current_lines = []
        current_audio_seg_ids = set()
        for i, line_words in lines_map.items():
            audio_seg_ids = set(line_words.audio_segment_ids)
            if not current_lines or not current_audio_seg_ids.isdisjoint(audio_seg_ids):
                current_lines.append(line_words)
                current_audio_seg_ids.update(audio_seg_ids)
            else:
                clusters.append((current_lines, current_audio_seg_ids))
                current_lines = [line_words]
                current_audio_seg_ids = audio_seg_ids
        if current_lines:
            clusters.append((current_lines, current_audio_seg_ids))
        # for validation
        audio_mask_ms = self._prepare_audiosegment_mask(audio_segments, 1.0)
        new_segments = []
        new_audio_segments = []
        for line_words, audio_seg_ids in clusters:
            line_words = list(chain.from_iterable(line_words))
            self._validate_line_to_audio(line_words, audio_seg_ids, audio_mask_ms, audio_segments)
            lang_list = [w.language for w in line_words if getattr(w, "language", None) is not None]
            new_segments.append(Segment(words=line_words, language=list(set(lang_list))))
            min_audio_start = audio_segments[min(audio_seg_ids)].start
            max_audio_end = audio_segments[max(audio_seg_ids)].end
            new_audio_segments.append(AudioSegment(start=float(round(min_audio_start, 3)), end=float(round(max_audio_end, 3))))
        assert len(new_segments) == len(new_audio_segments)
        if self.verbose:
            from rich import box  # type: ignore
            from rich.panel import Panel  # type: ignore
            from rich.table import Table  # type: ignore
            table = Table(title="Final Reference Timestamp", show_lines=True, box=box.MINIMAL,)
            table.add_column("IDX", style="cyan")
            table.add_column("Line Words", "blue")
            table.add_column("Language Hypothesis", "grey")
            table.add_column("Word Segment", style="magenta")
            table.add_column("New Audio Segment", style="green")
            table.add_column("Ori Audio Segment", style="green")
            prev_max_audio_id = -1
            for idx, ((seg, aseg), (line_words, audio_seg_ids)) in enumerate(zip(zip(new_segments, new_audio_segments), clusters)):
                min_audio_id, max_audio_id, = min(audio_seg_ids), max(audio_seg_ids)
                dropped_segment_between, dropped_segment_after = [], []
                for i in range(prev_max_audio_id + 1, min_audio_id): dropped_segment_between.append(audio_segments[i])
                prev_max_audio_id = max_audio_id
                if dropped_segment_between:
                    for drop in dropped_segment_between:
                        table.add_row("None", "None", "None", "None", "None",
                                        f"[red][{drop.h_start}-{drop.h_end}]\nDur: ({drop.duration:.3f})\nStatus: Dropped[/]")
                ori_lines = []
                for i in range(min_audio_id, max_audio_id + 1):
                    audio_seg = audio_segments[i]
                    if i in audio_seg_ids:
                        ori_lines.append(f"[{audio_seg.h_start}-{audio_seg.h_end}]")
                    else:
                        ori_lines.append(f"[red][{audio_seg.h_start}-{audio_seg.h_end}][/]")
                langs = ",".join(str(l) for l in seg.language) if getattr(seg, 'language', None) else "Unknown"
                table.add_row(str(idx), seg.text, langs,
                    f"[{seg.h_start}-{seg.h_end}]\nDur: ({seg.duration:.3f})",
                    f"[{aseg.h_start}-{aseg.h_end}]\nDur: ({aseg.duration:.3f})",
                    "\n".join(ori_lines))
                if idx == len(clusters) - 1 and prev_max_audio_id < len(audio_segments) - 1:
                    for i in range(prev_max_audio_id + 1, len(audio_segments)): dropped_segment_after.append(audio_segments[i])
                if dropped_segment_after:
                    for drop in dropped_segment_after:
                        table.add_row("None", "None", "None", "None", "None",
                                f"[red][{drop.h_start}-{drop.h_end}]\nDur: ({drop.duration:.3f})\nStatus: Dropped[/]")
            panel = Panel(table)
            self.console.print(panel)

        return new_segments, new_audio_segments

    def get_reference_timestamp(self,
        hypothesis: Result, reference: str,
        audio_segments: list[AudioSegment]
    ) -> tuple[Result, list[AudioSegment]]:
        ref_lines, ref_tokens, hyp_tokens = self.prepare_alignments(hypothesis, reference)
        aligned_map = self._sequence_align(ref_tokens, hyp_tokens)
        for m in aligned_map:
            if m.op in ["M", "S"]:
                lyric_token = ref_tokens[m.ref_idx]
                transcript_token = hyp_tokens[m.hyp_idx]
                lyric_token.start = float(round(transcript_token.start, 3))
                lyric_token.end = float(round(transcript_token.end, 3))
                lyric_token.score = float(transcript_token.score)
                lyric_token.language = str(transcript_token.language)
                lyric_token.source = "asr"
        lines_map = self._map_to_audio_segment(ref_tokens, audio_segments)
        final_segments, new_audio_segments = self._cluster_lines(lines_map, audio_segments)
        return Result(segments=final_segments), new_audio_segments
