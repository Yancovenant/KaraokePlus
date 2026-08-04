from __future__ import annotations

import difflib
import logging
import re
import string
from collections import defaultdict
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .aad import AudioSegment
from .transcriber import Result, Segment, WordTiming
from .utils import _process_audio

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    from .utils import AudioType


logger = logging.getLogger(__name__)

class AlignerAny:

    def __init__(self):
        self.is_qwen = False
        self.is_whisper = False

    def populate_model(self, model):
        self.sr = 16000 # qwen and whisper
        self.model = model
        self.is_whisper = model.__module__.startswith('faster_whisper.')
        self.is_qwen = model.__module__.startswith('qwen_asr.')

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
    
    def align(self,
              model,
              audio: AudioType,
              sr: int,
              result: Result,
              audio_segments: list[AudioSegment]) -> Result:
        if model is None:
            import copy
            self.get_default_model()
            is_default_model = True
        else:
            self.populate_model(model)
            is_default_model = False
        audio = _process_audio(audio, sr, self.sr)
        results = []
        desc = "Qwen" if self.is_qwen else ("Whisper" if self.is_whisper else "MMS_FA") + " Aligning..."
        with MainProgress(total = len(audio_segments), desc=desc, unit="chunk") as main_bar:
            try:
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
                        if self.is_qwen:
                            saved_safe_start.append(safe_start)
                            audio_chunk_list.append((audio_slice, self.sr))
                            text_chunk_list.append(res.text)
                            lang_chunk_list.append(res.language or "en") # Fallback currently to "en"
                        elif self.is_whisper:
                            align_results = self.model.align(audio_slice, res.text, verbose=None, language=res.language or "en") # Auto lang later on.
                            align_results = self.model.refine(audio_slice, align_results, steps="se", precision=0.02, verbose=None)
                            seg_words = []
                            for new_res in align_results.segments:
                                for word in new_res.words:
                                    seg_words.append(WordTiming(
                                        start=float(safe_start + word.start),
                                        end=float(safe_start + word.end),
                                        score=float(word.probability),
                                        word=str(word.word)
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
                                rw = copy.copy(res.words[i])
                                rw.start = local_start + safe_start
                                rw.end = local_end + safe_start
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


class ReferenceAligner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    _PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)

    def _sequence_align(self, reference: str | list[str], hypothesis: str | list[str]):
        env.sequence_align, env.pypinyin, env.pykakasi, env.anyascii  # noqa: B018
        from sequence_align.pairwise import needleman_wunsch_with_scores  # type: ignore  # noqa: I001
        from kplus.tools.romaji_converter import RomajiPhonetic
        @lru_cache(maxsize=2048)
        def get_phonetic(word: str):
            return RomajiPhonetic(word)
        def score_fn(a, b):
            if a == b: return 1.0 # Match exactly
            ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if ratio >= 0.6: return 1.0
            if get_phonetic(a) == get_phonetic(b): return 1.0
            return -3.0 # Mismatched
        if isinstance(reference, str): reference = [reference]
        if isinstance(hypothesis, str): hypothesis = [hypothesis]
        _ref, _hyp = needleman_wunsch_with_scores(
            reference, hypothesis, gap="-", score_fn=score_fn, indel_score=-1.0
        )
        hits = subs = dels = ins = 0
        aligned_map = []
        r_idx = h_idx = 0
        for i, (r_tok, h_tok) in enumerate(zip(_ref, _hyp)):
            if r_tok == "-": # insertation
                op="INST"; ins += 1
                aligned_map.append(SimpleNamespace(ref_idx=None, hyp_idx=h_idx, type=op))
                h_idx +=1; continue
            elif h_tok == "-": # deletation
                op="DELS"; dels += 1
                aligned_map.append(SimpleNamespace(ref_idx=r_idx, hyp_idx=None, type=op))
                r_idx +=1; continue
            elif r_tok == h_tok: # hit
                op="HITS"; hits += 1
            else: # subtitusion
                op="SUBS"; subs += 1
            aligned_map.append(SimpleNamespace(
                ref_idx=r_idx,
                hyp_idx=h_idx,
                type=op
            ))
            r_idx += 1; h_idx += 1

        if self.verbose:
            env.rich  # noqa: B018
            from rich.console import Console  # type: ignore
            console = Console()
            console.rule("Needleman Wunsch Alignment")
            console.print(
                f"[bold bright_green]Hits:[/bold bright_green] {hits} | "
                f"[bold bright_yellow]Subs:[/bold bright_yellow] {subs} | "
                f"[bold bright_red]Dels:[/bold bright_red] {dels} | "
                f"[bold bright_cyan]Ins:[/bold bright_cyan] {ins}\n"
            )
            colors = {"HITS": "grey70", "SUBS": "bold bright_yellow", "DELS": "bold bright_red", "INST": "bold bright_cyan",}
            cols = [
                (f"[{colors[m.type]}]{r:<{w}}[/]", f"[{colors[m.type]}]{h:<{w}}[/]", f"[{colors[m.type]}]{m.type:<{w}}[/]", w + 1)
                for r, h, m in zip(_ref, _hyp, aligned_map)
                for w in [max(len(str(r)), len(str(h)), len(m.type))]
            ]
            chunks, current_chunk, current_w = [], [], 0
            max_w = console.width - 5
            for *col, col_w in cols:
                if current_chunk and current_w + col_w > max_w:
                    chunks.append(current_chunk)
                    current_chunk, current_w = [], 0
                current_chunk.append(col)
                current_w += col_w
            if current_chunk:
                chunks.append(current_chunk)
            for chunk in chunks:
                for label, row in zip(["REF", "HYP", "OP "], zip(*chunk)):
                    console.print(f"{label}: " + " ".join(row))
                console.print("\n")
        return aligned_map

    def _line2segment(self,
        ref_lines_len   : int,                          # len(ref_lines)
        lines_map       : dict[int, list[WordTiming]],  # dict of line index and its corresponding words
        seg_index_map   : dict[AudioSegment, int],      # dictonary mapping, based on audio segment
        seg_midpoints   : list[float],                  # list of float for midpoint of audiosegments
        audio_segments  : list[AudioSegment]) -> dict[int, list[AudioSegment]]:
        """ Map lyric lines based on index to the corresponding
            audio segment
        """
        line_to_segments: dict[int, list[AudioSegment]] = {} # {index: list[AudioSegment]}
        for idx in range(ref_lines_len):
            line_words = lines_map[idx] # List[WordTiming]
            anchors = [w for w in line_words if w.start is not None] # List[WordTiming] >> if x.start != None
            if anchors:
                matched_segs = [] # List[AudioSegment]
                for w in anchors:
                    mid = (w.start + w.end) / 2
                    min_dist = float('inf')
                    closest_seg = audio_segments[0]
                    for i, seg in enumerate(audio_segments):
                        if seg.start <= mid <= seg.end: closest_seg = seg; break
                        dist = abs(mid - seg_midpoints[i]) # REVIEW: I think this will be able to be optimized
                        if dist < min_dist: min_dist = dist; closest_seg = seg
                    matched_segs.append(closest_seg)
                observed = sorted(set(matched_segs), key=lambda s: s.start)
                min_i = seg_index_map[observed[0]]
                max_i = seg_index_map[observed[-1]]
                if line_words[0].start is None:
                    prev_segs = line_to_segments.get(idx - 1)
                    prev_max_i = seg_index_map[prev_segs[-1]] if prev_segs else -1
                    if min_i - 1 > prev_max_i: min_i -= 1
                line_to_segments[idx] = audio_segments[min_i:max_i + 1]
            else:
                prev_segs = line_to_segments.get(idx - 1)
                line_to_segments[idx] = [prev_segs[-1] if prev_segs else audio_segments[0]]
        return line_to_segments

    def _2superlines(self,
        sorted_indices  : list[int],                    # from sorted(lines_map.keys())
        lines_map       : dict[int, list[WordTiming]],
        line_to_segments: dict[int, list[AudioSegment]],
        seg_index_map: dict[AudioSegment, int],
    ) -> list[tuple[list[WordTiming], AudioSegment, int, int]]: # (words, segment, min_idx, max_idx)
        """ Merge overlapping lines into 1 single audio segment and 1 single lines
        """
        super_phrases: list[tuple[list[WordTiming], AudioSegment, int, int]] = []
        current_words = lines_map[sorted_indices[0]] # The 1st line_maps list[WordTiming]
        current_segs = set(line_to_segments[sorted_indices[0]]) # The 1st list[AudioSegment] according to lines index
        for idx in sorted_indices[1:]:
            next_words = lines_map[idx] # The next list[WordTiming]
            next_segs = set(line_to_segments[idx]) # the next list[AudioSegment]
            if current_segs.isdisjoint(next_segs): # NO Overlapping, all this audio segments belong to the current segs
                min_start = min(s.start for s in current_segs)
                max_end = max(s.end for s in current_segs)
                min_idx = min(seg_index_map[s] for s in current_segs)
                max_idx = max(seg_index_map[s] for s in current_segs)
                super_phrases.append((
                    current_words, AudioSegment(start=min_start, end=max_end),
                    min_idx, max_idx
                ))
                current_words = next_words
                current_segs = next_segs
            else: # Overlap
                current_words.extend(next_words)
                current_segs.update(next_segs)
        min_start = min(s.start for s in current_segs)
        max_end = max(s.end for s in current_segs)
        min_idx = min(seg_index_map[s] for s in current_segs)
        max_idx = max(seg_index_map[s] for s in current_segs)
        super_phrases.append((
            current_words, AudioSegment(start=min_start, end=max_end),
            min_idx, max_idx
        ))
        return super_phrases

    def _interpolate_deleted_words(self,
        super_phrases: list[tuple[list[WordTiming], AudioSegment, int, int]], # (words, segment, min_idx, max_idx),
    ) -> tuple[list, list[AudioSegment]]:
        """ Interpolate any dropped words, and also map the new segment
        """
        final_segments, final_audio_segments = [], []
        for words, segment, _, _ in super_phrases:
            safe_start = segment.start
            safe_end = segment.end
            n = len(words)
            i = 0
            while i < n:
                if words[i].start is None:
                    block_start = i # Check inside this words list
                    while i < n and words[i].start is None:
                        i += 1
                    block_end = i
                    prev_end = safe_start
                    if block_start > 0:
                        logger.debug(f"{'':<4} Found leading end: {words[block_start - 1].word}-> {words[block_start - 1].end}")
                        prev_end = words[block_start - 1].end
                    next_start = safe_end
                    if block_end < n:
                        logger.debug(f"{'':<4} Found leading end: {words[block_end].word}-> {words[block_end].end}")
                        next_start = words[block_end].start

                    gap = next_start - prev_end
                    time_per_word = gap / (block_end - block_start)
                    curr_time = prev_end
                    for j in range(block_start, block_end):
                        logger.debug(f"{'':<8} Interpolate: [None - None] to [{curr_time:.2f}s - {curr_time + time_per_word:.2f}s]")
                        words[j].start = curr_time
                        words[j].end = curr_time + time_per_word
                        curr_time += time_per_word
                        words[j].source = "interpolated"
                else:
                    i += 1
            final_segments.append(Segment(words=words))
            final_audio_segments.append(segment)
        return final_segments, final_audio_segments

    @classmethod
    def normalize(self, s: str) -> str:
        s = s.translate(self._PUNCTUATION_TRANSLATOR).lower().strip()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s) # Kaldi
        return s
    
    def get_reference_timestamp(self, hypothesis: Result, reference: str, audio_segments: list[AudioSegment]) -> tuple[Result, list[AudioSegment]]:
        ref_lines = [line.strip() for line in reference.split("\n") if line.strip() and not line.startswith('[')]
        ref_tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            (token:=WordTiming(word=word, start=None, end=None, score=None),
                setattr(token, "line_idx", line_idx))[0]
            for line_idx, line in enumerate(ref_lines)
            for word in line.split()
        ]
        hyp_tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            WordTiming(word=word.word, start=word.start, end=word.end, score=word.score)
            for segment in hypothesis.segments
            for word in segment.words if word.score > 0.1
        ]
        ref_clean = [self.normalize(t.word) for t in ref_tokens]
        hyp_clean = [self.normalize(t.word) for t in hyp_tokens]
        aligned_map = self._sequence_align(ref_clean, hyp_clean)
        for match in aligned_map:
            if match.type in ["HITS", "SUBS"]:
                lyric_token = ref_tokens[match.ref_idx]
                transcript_token = hyp_tokens[match.hyp_idx]
                lyric_token.start = transcript_token.start
                lyric_token.end = transcript_token.end
                lyric_token.score = transcript_token.score
                lyric_token.source = "asr"
        lines_map: dict[int, list[WordTiming]] = defaultdict(list) # {line_idx: List[WordTiming]}
        for token in ref_tokens: lines_map[token.line_idx].append(token)
        seg_midpoints: list[float] = [(s.start + s.end) / 2 for s in audio_segments] # List[float, float, float]
        seg_index_map = {seg: i for i, seg in enumerate(audio_segments)} # {AudioSegment: index}
        line_to_segments: dict[int, list[AudioSegment]] = self._line2segment(len(ref_lines), lines_map, seg_index_map, seg_midpoints, audio_segments)

        sorted_indices = sorted(lines_map.keys())
        # (words, segment, min_idx, max_idx)
        super_phrases: list[tuple[list[WordTiming], AudioSegment, int, int]] = self._2superlines(sorted_indices, lines_map, line_to_segments, seg_index_map)
        final_segments, final_audio_segments = self._interpolate_deleted_words(super_phrases)
        if self.verbose:
            env.rich; from rich.console import Console; from rich.table import Table  # type: ignore  # noqa: B018, I001
            console = Console()
            console.rule("[bold bright_cyan]Reference Timestamps (Final State)[/]")
            for line_idx in sorted_indices:
                line_tokens = lines_map[line_idx]
                segs = line_to_segments.get(line_idx, [])
                if segs:
                    sorted_segs = sorted(segs, key=lambda s: s.start)
                    seg_indices_str = "<>".join(
                        str(seg_index_map[s]) for s in sorted_segs
                    )
                    audio_header = (
                        f"[bold grey70]{sorted_segs[0].h_start} -> {sorted_segs[-1].h_end}"
                        f" <= merged from [{seg_indices_str}][/]"
                    )
                else:
                    audio_header = "[bold grey70]No Audio Segments[/]"
                console.print(f"[bold bright_magenta]L{line_idx:02d}:[/] Audio: {audio_header}")
                words, stamps, durs = [], [], []
                for t in line_tokens:
                    source = getattr(t, "source", "asr")
                    if source == "asr": color = "bright_green"
                    elif source == "interpolated": color = "bright_yellow"
                    else: color = "bright_red"
                    ts = f"[{t.h_start}]" if t.start is not None else "[--:--.--]"
                    dur = f"({t.duration:.2f}s)" if t.start is not None else "(-.--s)"
                    col_w = max(len(t.word), len(ts), len(dur))
                    words.append(f"[bold bright_white]{t.word:<{col_w}}[/]")
                    stamps.append(f"[{color}]{ts:<{col_w}}[/]")
                    durs.append(f"[dim {color}]{dur:<{col_w}}[/]")
                console.print("[bold cyan]TEXT:[/] " + " ".join(words))
                console.print("[bold cyan]TS:  [/] " + " ".join(stamps))
                console.print("[bold cyan]DUR: [/] " + " ".join(durs))
                console.print("\n")
            table = Table(title="Final Merged Super Phrases", title_style="bold bright_cyan", header_style="bold dim", show_lines=True)
            table.add_column("IDX", justify="right", style="dim")
            table.add_column("Time Range", style="bright_green", justify="center")
            table.add_column("Dur", justify="right", style="bright_yellow")
            table.add_column("Phrase Transcript")
            prev_max_idx = -1
            for i, (words, seg, min_idx, max_idx) in enumerate(super_phrases):
                # Detect and render dropped audio segments between phrases
                if prev_max_idx != -1 and min_idx > prev_max_idx + 1:
                    dropped_count = min_idx - prev_max_idx - 1
                    drop_first = audio_segments[prev_max_idx + 1]
                    drop_last = audio_segments[min_idx - 1]
                    table.add_row(
                        "[bright_red]--[/]",
                        f"{drop_first.h_start} -> {drop_last.h_end}",
                        f"{drop_last.end - drop_first.start:.2f}s",
                        f"[bold bright_red]⚠ Dropped {dropped_count} segment(s)[/]",
                    )
                table.add_row(
                    str(i),
                    f"{seg.h_start} -> {seg.h_end}",
                    f"{seg.duration:.2f}s",
                    " ".join(w.word for w in words),
                )
                prev_max_idx = max_idx
            console.print(table)
            console.print("\n")
        return Result(segments=final_segments), final_audio_segments

