from __future__ import annotations

import logging
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .aad import AudioSegment
from .utils import Result, Segment, WordTiming, sec2ass

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType


logger = logging.getLogger(__name__)


class TranscriberMixin:
    def __init__(self, options):
        env.torchvision, env.qwen_asr # require torchvision first  # noqa: B018
        self.sr = 16000 # Whisper and qwen both used 16K sample rate
        self.verbose = options.verbose

    @classmethod
    def get_model(cls, options):
        if options.modeltype == "qwen":
            return QwenTranscribe(options)
        else:
            return WhisperTranscribe(options)

    def _process_audio(self, audio: AudioType, sr: int | None):
        env.numpy, env.torch, env.ffmpeg  # noqa: B018
        import numpy as np, torch  # type: ignore  # noqa: I001
        if isinstance(audio, torch.Tensor):
            from .utils import convert_audio
            assert sr is not None, "Passing ``torch.Tensor`` require to also have ``sr`` included"
            audio = convert_audio(audio, sr, self.sr, 1)
        elif isinstance(audio, (str, Path)):
            from .utils import load_audio
            audio: torch.Tensor = load_audio(audio, self.sr, 1)
        if not isinstance(audio, np.ndarray):
            audio = audio.detach().cpu().numpy().squeeze()
        return audio

    _PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)

    @classmethod
    def normalize(self, s: str) -> str:
        s = s.translate(TranscriberMixin._PUNCTUATION_TRANSLATOR).lower().strip()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s) # Kaldi
        return s

    def _sequence_align(self, reference: str | list[str], hypothesis: str | list[str]):
        env.sequence_align  # noqa: B018
        from sequence_align.pairwise import needleman_wunsch_with_scores  # type: ignore
        def score_fn(a, b): return 1.0 if a == b else -1.0
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
    
    def get_reference_timestamp(self, hypothesis: Result, reference: str, audio_segments: list[AudioSegment]):
        """ Synchronizes raw lyrics with audio transcripts and segments, 
            creating a temporally aligned result.
        """
        if not reference or not hypothesis.segments or not audio_segments: return Result(segments=[]), []
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

    def refine_timestamp(self, audio: AudioType, sr: int,
                         whisper_res: Result, qwen_res: Result,
                         audio_segments: list[AudioSegment]) -> Result:
        """ Head-to-head debug view of two forced aligners (whisper vs qwen),
        plus a midpoint consensus Result. Verdicts flag where the midpoint
        is a *guess* so you know exactly which words to scrub to. """
        env.librosa; import librosa  # type: ignore  # noqa: B018, I001
        from itertools import zip_longest  # noqa: F401
        audio_np = self._process_audio(audio, sr)
        total_dur = len(audio_np) / self.sr
        precision_ms=1
        hop_length = int(self.sr / 1000) * precision_ms # 16 frames
        frame_length = int(hop_length * 2) # 32 frames
        W_COLOR, Q_COLOR = "#0984e3", "#e84393"          # whisper=blue, qwen=magenta
        V_HEX = {"agree": "#2ecc71", "partial": "#f1c40f",
                "conflict": "#e74c3c", "mismatch": "#c0392b"}
        _PH = WordTiming(word="—", start=None, end=None, score=None)   # sentinel for zip_longest
        def _is_ph(w):  return w is _PH
        def _mid(a, b): return (a + b) / 2 if a is not None and b is not None else (a if a is not None else b)
        def _hms(w):    return "--:--.--" if (_is_ph(w) or w.start is None) else w.h_start
        def _hend(w):   return "--:--.--" if (_is_ph(w) or w.end   is None) else w.h_end

        if self.verbose:
            env.rich  # noqa: B018
            from rich.console import Console  # type: ignore
            from rich.panel import Panel  # type: ignore
            from rich.table import Table  # type: ignore
            from rich.text import Text  # type: ignore
            console = Console()
            console.rule("[bold cyan]⏱️ Acoustic Refinement & RMS Envelope Debug[/]")
        cons_segs: list[Segment] = []
        for i, (w_seg, q_seg, audio_seg) in enumerate(
                zip(whisper_res.segments, qwen_res.segments, audio_segments)):
            pairs = list(zip_longest(w_seg.words, q_seg.words, fillvalue=_PH))
            len_mismatch = len(w_seg.words) != len(q_seg.words)
            cons_words, meta = [], []
            for w1, w2 in pairs:
                s1, e1 = (None if _is_ph(w1) else w1.start, None if _is_ph(w1) else w1.end)
                s2, e2 = (None if _is_ph(w2) else w2.start, None if _is_ph(w2) else w2.end)
                d_start = (s2 - s1) * 1000 if (s1 is not None and s2 is not None) else None
                d_end   = (e2 - e1) * 1000 if (e1 is not None and e2 is not None) else None

                if _is_ph(w1) or _is_ph(w2):
                    verdict = "mismatch"
                else:
                    worst = max(abs(d_start), abs(d_end))
                    verdict = "agree" if worst < 50 else ("conflict" if worst > 150 else "partial")

                cons_start, cons_end = _mid(s1, s2), _mid(e1, e2)
                if cons_end is not None and cons_start is not None and cons_end < cons_start:
                    cons_end = cons_start + 0.03

                word_text = w1.word if not _is_ph(w1) else w2.word
                cw = WordTiming(word=word_text, start=cons_start, end=cons_end,
                                score=getattr(w1, "score", None) if not _is_ph(w1) else getattr(w2, "score", None))
                cw.verdict = verdict                                   # stash trust on the word itself
                for attr in ("line_idx", "source"):
                    src = w1 if not _is_ph(w1) else w2
                    if hasattr(src, attr): setattr(cw, attr, getattr(src, attr))
                cons_words.append(cw)
                ss = [x for x in (s1, s2) if x is not None]
                meta.append({
                    "word": word_text, "verdict": verdict, "color": V_HEX[verdict],
                    "w1s": s1, "w2s": s2, "w1e": e1, "w2e": e2, "d_start": d_start, "d_end": d_end,
                    "w1_hs": _hms(w1), "w2_hs": _hms(w2), "w1_he": _hend(w1), "w2_he": _hend(w2),
                    "cons_start": cons_start, "cons_end": cons_end,
                    "min_s": min(ss) if ss else cons_start, "max_s": max(ss) if ss else cons_start,
                })
            cons_segs.append(Segment(words=cons_words))
            a_starts = [x for x in (w_seg.start, q_seg.start) if x is not None] or [0.0]
            a_ends   = [x for x in (w_seg.end,   q_seg.end)   if x is not None] or [0.0]
            earliest, latest = min(a_starts), max(a_ends)
            inner_s = min(earliest, audio_seg.start) if audio_seg.start is not None else earliest
            inner_e = max(latest,   audio_seg.end)   if audio_seg.end   is not None else latest
            safe_start = max(0.0, inner_s, earliest - 1.0)
            safe_end   = min(total_dur, inner_e, latest + 1.0)
            s0, s1 = int(safe_start * self.sr), int(safe_end * self.sr)
            audio_chunk = audio_np[s0:s1]
            if len(audio_chunk) == 0:
                console.print(f"[bright_yellow]⚠ Empty audio chunk for Segment {i:02d}. Skipping.[/]"); continue

            fl = min(frame_length, len(audio_chunk)) or 1
            hl = min(hop_length,   len(audio_chunk)) or 1
            rms = librosa.feature.rms(y=audio_chunk, frame_length=fl, hop_length=hl)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=hl) + safe_start

            # --- header with a verdict tally ---
            n_agree  = sum(m["verdict"] == "agree"     for m in meta)
            n_part   = sum(m["verdict"] == "partial"   for m in meta)
            n_conf   = sum(m["verdict"] == "conflict"  for m in meta)
            n_mis    = sum(m["verdict"] == "mismatch"  for m in meta)
            console.print(Panel(Text.from_markup(
                f"[bold]Segment {i:02d}[/]  audio [cyan]{audio_seg.h_start} → {audio_seg.h_end}[/]\n"
                f"[bright_blue]whisper[/] [dim]{w_seg.h_start} → {w_seg.h_end}[/]   "
                f"[bright_magenta]qwen[/] [dim]{q_seg.h_start} → {q_seg.h_end}[/]\n"
                f"verdicts: [bright_green]{n_agree} agree[/] · [bright_yellow]{n_part} partial[/] · "
                f"[bright_red]{n_conf} conflict[/] · [bold bright_red]{n_mis} mismatch[/]"
            ), border_style="cyan", expand=False))
            if len_mismatch:
                console.print(f"[bold bright_red]⚠ Word-count mismatch: whisper={len(w_seg.words)} "
                            f"qwen={len(q_seg.words)} — lanes/Δ are positionally padded, NOT semantically aligned.[/]")

            # --- word table: one column per witness + Δ + verdict ---
            def _dcell(d):
                if d is None: return "[bold bright_red]miss[/]"
                c = "bright_green" if abs(d) < 50 else ("bright_yellow" if abs(d) < 150 else "bright_red")
                return f"[{c}]{d:+.1f}ms[/]"

            wt = Table(title=f"Word-Level (Segment {i:02d})", show_lines=False, expand=True, border_style="dim")
            wt.add_column("Word", style="bold white", no_wrap=True)
            wt.add_column("whisper▶", justify="center", style="bright_blue")
            wt.add_column("qwen▶",    justify="center", style="bright_magenta")
            wt.add_column("Δ start",  justify="right")
            wt.add_column("whisper◀", justify="center", style="bright_blue")
            wt.add_column("qwen◀",    justify="center", style="bright_magenta")
            wt.add_column("Δ end",    justify="right")
            wt.add_column("verdict",  justify="center")
            for m in meta:
                wt.add_row(
                    escape(m["word"]),
                    m["w1_hs"], m["w2_hs"], _dcell(m["d_start"]),
                    m["w1_he"], m["w2_he"], _dcell(m["d_end"]),
                    f"[{('bold ' if m['verdict']=='mismatch' else '')}{_dcell(0).split('[')[1] if False else ''}]"
                    f"{'[bright_green]agree[/]' if m['verdict']=='agree' else '[bright_yellow]partial[/]' if m['verdict']=='partial' else '[bright_red]conflict[/]' if m['verdict']=='conflict' else '[bold bright_red]mismatch[/]'}",
                )
            console.print(wt)
            if not sys.stdout.isatty():
                env.plotly; import base64  # noqa: B018, I001
                import plotly.graph_objects as go  # type: ignore
                from IPython.display import HTML, Audio, display  # type: ignore
                from plotly.subplots import make_subplots  # type: ignore

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.34, 0.26, 0.40],
                    subplot_titles=("Raw Waveform", "RMS Energy (dB)", "Witness Lanes → Verdict"))
                time_axis = np.linspace(safe_start, safe_end, len(audio_chunk))
                fig.add_trace(go.Scatter(x=time_axis, y=audio_chunk, name="Waveform",
                                        line={"color": "#00d2ff", "width": 1}, showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=rms_times, y=rms_db, name="RMS (dB)", fill="tozeroy",
                                        line={"color": "#ffaa00", "width": 2}, showlegend=False), row=2, col=1)

                # one toggleable Gantt lane per witness (built from each witness's OWN word list)
                for name, color, words in (("whisper", W_COLOR, w_seg.words),
                                        ("qwen",    Q_COLOR, q_seg.words)):
                    ws = [w for w in words if w.start is not None and w.end is not None]
                    if not ws: continue
                    fig.add_trace(go.Bar(
                        orientation="h", y=[name] * len(ws),
                        base=[w.start for w in ws],
                        x=[max(0.02, w.end - w.start) for w in ws],
                        text=[w.word for w in ws], textposition="inside", insidetextanchor="middle",
                        textfont={"color": "white", "size": 9}, cliponaxis=False,
                        marker_color=color, opacity=0.55, name=name, legendgroup=name,
                        hovertemplate=f"<b>{name}</b> | %{{text}}<br>start %{{base:.3f}}s<extra></extra>",
                    ), row=3, col=1)

                # the VERDICT lane = consensus chips, colored by agreement
                vm = [m for m in meta if m["cons_start"] is not None and m["cons_end"] is not None]
                if vm:
                    fig.add_trace(go.Bar(
                        orientation="h", y=["VERDICT"] * len(vm),
                        base=[m["cons_start"] for m in vm],
                        x=[max(0.02, m["cons_end"] - m["cons_start"]) for m in vm],
                        text=[m["word"] for m in vm], textposition="inside", insidetextanchor="middle",
                        textfont={"color": "white", "size": 9}, cliponaxis=False,
                        marker_color=[m["color"] for m in vm], opacity=0.95,
                        name="VERDICT", legendgroup="VERDICT",
                        customdata=[m["verdict"] for m in vm],
                        hovertemplate="<b>VERDICT</b> | %{text} [%{customdata}]<br>start %{base:.3f}s<extra></extra>",
                    ), row=3, col=1)
                fig.update_yaxes(type="category", categoryorder="array",
                                categoryarray=["whisper", "qwen", "VERDICT"], row=3, col=1)
                fig.update_layout(bargap=0.25)

                # conflict bands: shade the start-disagreement zone across ALL rows
                for m in meta:
                    if m["verdict"] != "agree" and m["min_s"] != m["max_s"]:
                        fig.add_vrect(x0=m["min_s"], x1=m["max_s"], fillcolor="#e74c3c",
                                    opacity=0.12, line_width=0, layer="below", row="all")
                # consensus decision guides (tie audio ↔ lanes)
                for m in meta:
                    if m["cons_start"] is not None:
                        fig.add_vline(x=m["cons_start"], line_width=1, line_color="#9b59b6", opacity=0.6, row="all")
                # segment boundaries: witnesses dashed, consensus solid
                fig.add_vrect(x0=cons_segs[-1].start, x1=cons_segs[-1].end, fillcolor="green",
                            opacity=0.10, layer="below", line_width=0, row="all")
                fig.add_vline(x=cons_segs[-1].start, line_width=2, line_color="green", row="all")
                fig.add_vline(x=cons_segs[-1].end,   line_width=2, line_color="red",   row="all")
                fig.add_vline(x=w_seg.start, line_width=1, line_dash="dash", line_color=W_COLOR, row="all")
                fig.add_vline(x=q_seg.start, line_width=1, line_dash="dash", line_color=Q_COLOR, row="all")

                fig.update_layout(template="plotly_dark", hovermode="x unified", height=640,
                                margin={"l": 20, "r": 20, "t": 40, "b": 20}, showlegend=True)
                fig.show()

                b64 = base64.b64encode(fig.to_html(include_plotlyjs="cdn", full_html=True).encode()).decode()
                display(HTML(f"""
                    <a href="data:text/html;base64,{b64}" download="False" target="_blank" style="display:block;
                    text-align:center; text-decoration:none; padding:14px 24px; font-size:16px; font-weight:bold;
                    background-color:#00d2ff; color:black; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,.3);
                    margin:10px 0; cursor:pointer;">Open Graph in Full Tab / Download Graph</a>"""))
                display(Audio(data=audio_chunk, rate=self.sr, autoplay=False))
                console.print()
        return
        for i, (first_res, second_res, audio_seg) in enumerate(zip(whisper_res.segments, qwen_res.segments, audio_segments)):
            min_start = min(0, first_res.start, second_res.start)
            max_end = max(first_res.end, second_res.end)
            safe_start = max(min(min_start, audio_seg.start), min_start - 1.0)
            safe_end = min(max(max_end, audio_seg.end), max_end + 1.0)
            start, end = int(safe_start*self.sr), int(safe_end*self.sr)
            audio_chunk = audio_np[start:end]
            
            rms = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)[0]
            if self.verbose:
                if len(audio_chunk) == 0: console.print(f"[bright_yellow]⚠ Empty audio chunk for Segment {i:02d}. Skipping.[/]"); continue
                header = Text.from_markup(
                    f"[bold]AudioSegment {i:02d}: [{audio_seg.h_start} - {audio_seg.h_end}][/]"
                    f"1st: [{first_res.h_start} - {first_res.h_end}]"
                    f"2nd: [{second_res.h_start} - {second_res.h_end}]"
                )
                console.print(Panel(header, border_style="cyan", expand=False))
                word_table = Table(title=f"Word-Level (Segment {i:02d})", show_lines=False, expand=True, border_style="dim")
                word_table.add_column("Word", style="bold white", no_wrap=True)
                word_table.add_column("1st Start", justify="center", style="grey70")
                word_table.add_column("2nd Start", justify="center", style="bright_cyan")
                word_table.add_column("Δ Start", justify="right")
                word_table.add_column("Orig End", justify="center", style="grey70")
                word_table.add_column("Ref End", justify="center", style="bright_cyan")
                word_table.add_column("Δ End", justify="right")
                for first_res_word, second_res_word in zip(first_res.words, second_res.words):
                    d_start = (second_res_word.start - first_res_word.start) * 1000
                    d_end = (second_res_word.end - first_res_word.end) * 1000
                    c_start = "bright_green" if abs(d_start) < 50 else ("bright_yellow" if abs(d_start) < 150 else "bright_red")
                    c_end = "bright_green" if abs(d_end) < 50 else ("bright_yellow" if abs(d_end) < 150 else "bright_red")
                    
                    word_table.add_row(
                        first_res_word.word,
                        first_res_word.h_start, second_res_word.h_start, f"[{c_start}]{d_start:+.1f}ms[/]",
                        first_res_word.h_end, second_res_word.h_end, f"[{c_end}]{d_end:+.1f}ms[/]"
                    )
                console.print(word_table)
                rms_db = librosa.amplitude_to_db(rms, ref=np.max)
                rms_times = librosa.frames_to_time(np.arange(len(rms)),sr=self.sr, hop_length=hop_length) + safe_start
                if not sys.stdout.isatty():
                    env.plotly; import base64  # noqa: B018, I001
                    import plotly.graph_objects as go # type: ignore
                    from IPython.display import HTML, Audio, display # type: ignore
                    from plotly.subplots import make_subplots  # type: ignore
                    fig = make_subplots(
                        rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        row_heights=[0.5],
                        subplot_titles=("Raw Waveform", "RMS Energy (dB)"),
                    )
                    time_axis = np.linspace(safe_start, safe_end, len(audio_chunk))
                    fig.add_trace(go.Scatter(x=time_axis, y=audio_chunk, name="Waveform",
                                                line={"color": "#00d2ff", "width": 1}), row=1, col=1)
                    for first_res_word, second_res_word in zip(first_res.words, second_res.words):
                        fig.add_vrect(x0=first_res_word.start, x1=first_res_word.end, fillcolor="#2ecc71", row=1, opacity=0.30)
                        fig.add_vrect(x0=second_res_word.start, x1=second_res_word.end, fillcolor="#2ecc71", row=1, opacity=0.30)
                        fig.add_vline(x=first_res_word.start, line_width=2, line_color="#f1c40f", name="1st Word start")
                        fig.add_vline(x=first_res_word.end, line_width=2, line_color="#f1c40f", name="1st Word end")
                        fig.add_vline(x=second_res_word.start, line_width=2, line_color="#f1c40f", name="2nd Word start")
                        fig.add_vline(x=second_res_word.end, line_width=2, line_color="#f1c40f", name="2nd Word end")
                        fig.add_annotation(
                            xref="x", yref="y2",
                            x=(first_res_word.start + first_res_word.end) / 2, y=0.25,
                            text=first_res_word.word, showarrow=False,
                            font={"color": "white", "size": 11},
                            bgcolor="#e74c3c", opacity=0.92,
                        )
                        fig.add_annotation(
                            xref="x", yref="y2",
                            x=(second_res_word.start + second_res_word.end) / 2, y=0.75,
                            text=second_res_word.word, showarrow=False,
                            font={"color": "white", "size": 11},
                            bgcolor="#e74c3c", opacity=0.92,
                        )
                    fig.update_layout(
                        template="plotly_dark",
                        hovermode="x unified",
                        height=450,
                        margin={"l": 20, "r": 20, "t": 40, "b": 20},
                        showlegend=False
                    )
                    fig.show()
        return
        for i, (prev_res, new_res, audio_seg) in enumerate(zip(ref_result.segments, align_results.segments, audio_segments)):
            safe_start = max(min(new_res.start, audio_seg.start), new_res.start - 0.8)
            safe_end = min(max(new_res.end, audio_seg.end), new_res.end + 1.5)
            start, end = int(safe_start*self.sr), int(safe_end*self.sr)
            audio_chunk = audio_np[start:end]
            hop_length = int(self.sr / 1000) * precision_ms # 16 frames
            frame_length = int(hop_length * 2) # 32 frames
            rms = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)[0]
            if self.verbose:
                if len(audio_chunk) == 0:
                    console.print(f"[bright_yellow]⚠ Empty audio chunk for Segment {i:02d}. Skipping.[/]")
                    continue
                delta_start_ms = (new_res.start - prev_res.start) * 1000
                delta_end_ms = (new_res.end - prev_res.end) * 1000
                shift_start_color = "bright_green" if abs(delta_start_ms) < 50 else ("bright_yellow" if abs(delta_start_ms) < 150 else "bright_red")
                shift_end_color = "bright_green" if abs(delta_end_ms) < 50 else ("bright_yellow" if abs(delta_end_ms) < 150 else "bright_red")
                header = Text.from_markup(
                    f"[bold]Segment {i:02d}[/] | Duration: [{audio_seg.h_start} - {audio_seg.h_end}]\n"
                    f"Refined: [cyan]{new_res.h_start} → {new_res.h_end}[/]\n"
                    f"Original: [dim]{prev_res.h_start} → {prev_res.h_end}[/]\n"
                    f"Δ Shift: Start [{shift_start_color}]{delta_start_ms:+.1f}ms[/] | "
                    f"End [{shift_end_color}]{delta_end_ms:+.1f}ms[/]"
                )
                console.print(Panel(header, border_style="cyan", expand=False))
                # Word
                word_table = Table(title=f"Word-Level Shifts (Segment {i:02d})", show_lines=False, expand=True, border_style="dim")
                word_table.add_column("Word", style="bold white", no_wrap=True)
                word_table.add_column("Orig Start", justify="center", style="dim")
                word_table.add_column("Ref Start", justify="center", style="cyan")
                word_table.add_column("Δ Start", justify="right")
                word_table.add_column("Orig End", justify="center", style="dim")
                word_table.add_column("Ref End", justify="center", style="cyan")
                word_table.add_column("Δ End", justify="right")
                for prev_word, new_word in zip(prev_res.words, new_res.words):
                    d_start = (new_word.start - prev_word.start) * 1000
                    d_end = (new_word.end - prev_word.end) * 1000
                    c_start = "bright_green" if abs(d_start) < 50 else ("bright_yellow" if abs(d_start) < 150 else "bright_red")
                    c_end = "bright_green" if abs(d_end) < 50 else ("bright_yellow" if abs(d_end) < 150 else "bright_red")
                    
                    word_table.add_row(
                        prev_word.word,
                        prev_word.h_start, new_word.h_start, f"[{c_start}]{d_start:+.1f}ms[/]",
                        prev_word.h_end, new_word.h_end, f"[{c_end}]{d_end:+.1f}ms[/]"
                    )
                console.print(word_table)
                rms_db = librosa.amplitude_to_db(rms, ref=np.max)
                rms_times = librosa.frames_to_time(np.arange(len(rms)),sr=self.sr, hop_length=hop_length) + safe_start
                ## ASS Video Preview
                full_ass = Result.ASS_HEADER + new_res.ass_event
                if sys.stdout.isatty():
                    env.plotext; import plotext as plt  # type: ignore  # noqa: B018, I001
                    plt.clear_figure()
                    plt.theme("pro")
                    plt.plot(rms_times, rms_db, label="RMS Energy (dB)", color="yellow", marker="hd")
                    plt.vline(new_res.start, color="green")
                    plt.vline(new_res.end, color="red")
                    plt.vline(prev_res.start, color="cyan")
                    plt.vline(prev_res.end, color="magenta")
                    plt.title(f"Segment {i:02d} RMS Envelope & Boundaries")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude (dB)")
                    plt.show()
                    console.print()
                else:
                    env.plotly; import json, base64  # noqa: B018, I001
                    import plotly.graph_objects as go # type: ignore
                    from IPython.display import HTML, Audio, display # type: ignore
                    from plotly.subplots import make_subplots  # type: ignore
                    def _shift_hex(ms: float) -> str:
                        a = abs(ms)
                        return "#2ecc71" if a < 50 else ("#f1c40f" if a < 150 else "#e74c3c")
                    fig = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        row_heights=[0.5, 0.5],
                        subplot_titles=("Raw Waveform", "RMS Energy (dB)"),
                    )
                    time_axis = np.linspace(safe_start, safe_end, len(audio_chunk))
                    fig.add_trace(go.Scatter(x=time_axis, y=audio_chunk, name="Waveform",
                                             line={"color": "#00d2ff", "width": 1}), row=1, col=1)
                    fig.add_trace(go.Scatter(x=rms_times, y=rms_db, name="RMS (dB)",
                                             fill="tozeroy", line={"color": "#ffaa00", "width": 2}), row=2, col=1)
                    for word in new_res.words:
                        fig.add_vrect(x0=word.start, x1=word.end, fillcolor="#2ecc71", row="all", opacity=0.30)
                        fig.add_vline(x=word.start, line_width=2, line_color="#f1c40f", name="Word start")
                        fig.add_vline(x=word.end, line_width=2, line_color="#f1c40f", name="Word end")
                        fig.add_annotation(
                            xref="x", yref="y2",
                            x=(word.start + word.end) / 2, y=0.5,
                            text=word.word, showarrow=False,
                            font={"color": "white", "size": 11},
                            bgcolor="#e74c3c", opacity=0.92,
                        )
                    fig.add_vrect(x0=new_res.start, x1=new_res.end, fillcolor="green", opacity=0.15,
                                  layer="below", line_width=0, row="all")
                    fig.add_vline(x=new_res.start, line_width=2, line_color="green", name="Refined Start", row="all")
                    fig.add_vline(x=new_res.end, line_width=2, line_color="red", name="Refined End", row="all")
                    fig.add_vline(x=prev_res.start, line_width=1, line_dash="dash", line_color="cyan", name="Original Start", row="all")
                    fig.add_vline(x=prev_res.end, line_width=1, line_dash="dash", line_color="magenta", name="Original End", row="all")
                    fig.update_layout(
                        template="plotly_dark",
                        hovermode="x unified",
                        height=450,
                        margin={"l": 20, "r": 20, "t": 40, "b": 20},
                        showlegend=False
                    )
                    fig.show()
                    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
                    b64_html = base64.b64encode(html_str.encode('utf-8')).decode('utf-8')
                    button_html = f"""
                    <a href="data:text/html;base64,{b64_html}" download="plotly_graph.html"  target="_blank" style="
                        display: block;
                        text-align: center;
                        text-decoration: none;
                        padding: 14px 24px;
                        font-size: 16px;
                        font-weight: bold;
                        background-color: #00d2ff;
                        color: black;
                        border-radius: 8px;
                        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
                        margin: 10px 0;
                        cursor: pointer;
                    ">
                        Open Graph in Full Tab / Download Graph
                    </a>
                    """
                    display(HTML(button_html))
                    display(Audio(data=audio_chunk, rate=self.sr, autoplay=False))
                    self._render_ass_preview(audio_np, new_res)
                    console.print()

    def _render_ass_preview(self, audio_np, segment: Segment):
        """ Renders an ASS video preview using FFmpeg and libass. """
        env.rich, env.ffmpeg  # noqa: B018
        import rich, soundfile as sf  # type: ignore # noqa: I001
        console = rich.console.Console()
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = Path(tmpdir) / "audio.wav"
                ass_path = Path(tmpdir) / "subs.ass"
                output_path = Path(tmpdir) / "preview.mp4"
                parts = segment.ass_event.split(',', 9)
                def ass2sec(t):
                    h, m, s = t.split(':')
                    return int(h)*3600 + int(m)*60 + float(s)
                if len(parts) >= 10 and parts[0].startswith("Dialogue"):
                    start_sec = ass2sec(parts[1])
                    end_sec = ass2sec(parts[2])
                    duration = end_sec - start_sec
                    # safe start is max(min(new_res.start, audio_seg.start), new_res.start - 0.8)
                    # While our ass event is pad_start = max(0.0, current.start - 0.8, prev_end)
                    safe_start = int(start_sec * self.sr)
                    safe_end = int(end_sec * self.sr)
                    safe_start = max(0, min(safe_start, len(audio_np)))
                    safe_end = max(safe_start, min(safe_end, len(audio_np)))
                    audio_chunk = audio_np[safe_start:safe_end]
                    parts[1] = sec2ass(0.0)
                    parts[2] = sec2ass(duration)
                    shifted_event = ",".join(parts)
                else:
                    shifted_event = segment.ass_event
                    audio_chunk = audio_np
                sf.write(audio_path, audio_chunk, self.sr)
                console.print(f"ASS EVENT:\n{shifted_event}")
                ass_content = Result.ASS_HEADER + shifted_event + "\n"
                with open(ass_path, "w", encoding="utf-8-sig") as f:
                    f.write(ass_content)
                duration = len(audio_chunk) / self.sr
                ass_path_ffmpeg = ass_path.as_posix()
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=black:s=1280x720:r=30:d={duration}",
                    "-i", audio_path,
                    "-vf", f"ass={ass_path_ffmpeg}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k",
                    "-shortest",
                    output_path
                ]
                import subprocess
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0:
                    from IPython.display import display, Video  # type: ignore # noqa: I001
                    # embed=True base64 encodes the MP4 so it displays perfectly in Jupyter
                    display(Video(output_path, embed=True, width=720))
                else:
                    console.print(f"[red]⚠ FFmpeg failed to render preview: {result.stderr.decode()[:150]}[/]")

        except Exception as e:
            console.print(f"[yellow]⚠ Video preview skipped: {e}[/]")
            


    def transcribe(self, audio: AudioType, audio_segments: list[AudioSegment] | None = None, lyrics: str | None = None, sr: int | None = None) -> Result:
        raise NotImplementedError()

    def align(self, audio: AudioType, sr: int, result: Result, audio_segments: list[AudioSegment]) -> Result:
        raise NotImplementedError()


class QwenTranscribe(TranscriberMixin):
    def __init__(self, options):
        super().__init__(options)
        env.qwen_asr  # noqa: B018
        from qwen_asr import Qwen3ASRModel  # type: ignore # noqa: I001
        import torch # type: ignore
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.float16,
            device_map=env.device.type,
            attn_implementation="sdpa",
            max_inference_batch_size=-1, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=4096, # Maximum number of tokens to generate. Set a larger value for long audio input.
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs={"dtype": torch.float16,
                                   "device_map": env.device.type,
                                   "attn_implementation": "sdpa",},)

    def _apply_model(self, audio_segments: list[AudioSegment], audio: list[tuple[np.ndarray, int]], context: list[str] | None = None):
        results = []
        with MainProgress(total = len(audio), desc="Qwen Starting Transcriptions...", unit="chunk") as main_bar:
            try:
                batch_result =  self.model.transcribe(audio=audio, context=context, return_time_stamps=True,)
                for seg, aseg in zip(batch_result, audio_segments):
                    main_bar.update(1)
                    if seg.time_stamps is not None:
                        seg_words = []
                        for word in seg.time_stamps:
                            seg_words.append(WordTiming(
                                start=float(word.start_time + aseg.start),
                                end=float(word.end_time + aseg.start),
                                score=1.0,
                                word=str(word.text)
                            ))
                        results.append(Segment(words=seg_words, language=seg.language))
            except Exception:
                logger.exception("!!! Whisper Transcription Error:")
                raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        logger.debug(">> Qwen Transcription:")
        for res in results:
            logger.debug(f"{'':<2}Segment: {res.start:.2f}s - {res.end:.2f}s (Duration: {res.end-res.start:.2f}s) {res.text}")
            for w in res.words:
                logger.debug(f"{'':<4}WordTiming: ({w.score:.2f}) {w.start:.2f}s to {w.end:.2f}s {w.word}")
        env.clean()
        return results

    def transcribe(self, audio: AudioType, audio_segments: list[AudioSegment] | None = None, lyrics: str | None = None, sr: int | None = None) -> Result:
        audio = self._process_audio(audio, sr)
        if not audio_segments:
            duration = len(audio) / self.sr
            audio_segments = [AudioSegment(start=0.0, end=duration)]
        audio_chunk_list = []
        for seg in audio_segments:
            start, end = int(seg.start * self.sr), int(seg.end * self.sr)
            audio_chunk_list.append((audio[start:end], self.sr))
        results = self._apply_model(audio_segments=audio_segments, audio=audio_chunk_list)
        return Result(segments=results)

    def align(self, audio: AudioType, sr: int, result: Result, audio_segments: list[AudioSegment]) -> Result:
        audio = self._process_audio(audio, sr)
        audio_chunk_list, text_chunk_list, lang_chunk_list, saved_safe_start = [], [], [], []
        for res, seg in zip(result.segments, audio_segments):
            if seg.start <= ((res.end + res.start) / 2) <= seg.end:
                # Capped maximum 1s
                safe_start = max(min(res.start, seg.start), res.start - 1.0)
                safe_end = min(max(res.end, seg.end), res.end + 1.0)
                saved_safe_start.append(safe_start)
                start_sample = int(safe_start * self.sr)
                end_sample = int(safe_end * self.sr)
                audio_slice = audio[start_sample:end_sample]
                assert audio_slice.shape[0] > 0
                audio_chunk_list.append((audio_slice, self.sr))
                text_chunk_list.append(res.text)
                lang_chunk_list.append(res.language or "en") # Fallback currently to "en"
        align_results = self.model.forced_aligner.align(
            audio=audio_chunk_list, text=text_chunk_list, language=lang_chunk_list,
        )
        results = []
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
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        env.clean()
        return Result(segments=results)


class WhisperTranscribe(TranscriberMixin):
    def __init__(self, options):
        super().__init__(options)
        env.stable_ts, env.faster_whisper  # noqa: B018
        import stable_whisper  # type: ignore
        self.max_threads = max(1, options.max_threads)
        self.beamsize = options.beamsize
        compute_type = "float16" if env.device.type == "cuda" else "float32"
        self.model = stable_whisper.load_faster_whisper(options.modelname, device=env.device.type, compute_type=compute_type, num_workers=options.max_threads)

    def transcribe(self, audio: AudioType, audio_segments: list[AudioSegment] | None = None, lyrics: str | None = None, sr: int | None = None) -> Result:
        audio = self._process_audio(audio, sr)
        if not audio_segments:
            duration = len(audio) / self.sr
            audio_segments = [AudioSegment(start=0.0, end=duration)]
        results = []
        with MainProgress(total = len(audio_segments), desc="Whisper Starting Transcriptions...", unit="chunk") as main_bar:
            try:
                # normal transcribe expect str, float() timestamp
                time_batches = ",".join(f"{seg.start},{seg.end}" for seg in audio_segments)
                # batch transcribe expect a dict?
                # time_batches = [{"start": seg.start, "end": seg.end} for seg in audio_segments]
                batch_result = self.model.transcribe(audio, language=None, clip_timestamps=time_batches,
                                                    initial_prompt=lyrics, beam_size=self.beamsize, #batch_size=1, # need to be None
                                                    repetition_penalty=1.2, condition_on_previous_text=False)
                for res in batch_result:
                    main_bar.update(1)
                    seg_words = []
                    for w in res.words:
                        seg_words.append(WordTiming(
                            start=float(w.start),
                            end=float(w.end),
                            score=float(w.probability),
                            word=str(w.word)))
                    results.append(Segment(words=seg_words, language=batch_result.language))
            except Exception:
                logger.exception("!!! Whisper Transcription Error:")
                raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        logger.debug(">> Whisper Transcription:")
        for res in results:
            logger.debug(f"{'':<2}Segment: {res.start:.2f}s - {res.end:.2f}s (Duration: {res.end-res.start:.2f}s)")
            for w in res.words:
                logger.debug(f"{'':<4}WordTiming: ({w.score:.2f}) {w.start:.2f}s to {w.end:.2f}s {w.word}")
        env.clean()
        return Result(segments=results)
     
    def align(self, audio: AudioType, sr: int, result: Result, audio_segments: list[AudioSegment]) -> Result:
        audio = self._process_audio(audio, sr)
        results = []
        with MainProgress(total = len(audio_segments), desc="Whisper Starting align + refine...", unit="chunk") as main_bar:
            for res, seg in zip(result.segments, audio_segments):
                if seg.start <= ((res.end + res.start) / 2) <= seg.end:
                    safe_start = max(min(res.start, seg.start), res.start - 1.0)
                    safe_end = min(max(res.end, seg.end), res.end + 1.0)
                    start_sample = int(safe_start * self.sr)
                    end_sample = int(safe_end * self.sr)
                    audio_slice = audio[start_sample:end_sample]
                    assert audio_slice.shape[0] > 0
                    try:
                        align_results = self.model.align(audio_slice, res.text, verbose=None, language=res.language or "en") # Auto lang later on.
                        align_results = self.model.refine(audio_slice, align_results, steps="ss", precision=0.5, verbose=None)
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
                    except Exception as err:
                        logger.exception(f"!!! Whisper align + refine Error: {err}")
                        raise
            results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
            env.clean()
            return Result(segments=results)
