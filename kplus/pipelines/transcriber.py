from __future__ import annotations

import logging
import re
import string
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .aad import AudioSegment
from .utils import Result, Segment, WordTiming

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

    def _plot_jiwer(self, ref: str, res: str) -> None:
        env.jiwer  # noqa: B018
        import jiwer  # type: ignore
        res_flat = " ".join([seg.text for seg in res.segments])
        ref_flat = " ".join([line.strip() for line in ref.split("\n") if line.strip() and not line.startswith('[')])
        normalizer = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        out = jiwer.process_words(
            ref_flat,
            res_flat,
            reference_transform=normalizer,
            hypothesis_transform=normalizer
        )
        logger.info("--- Transcription Metrics ---")
        metrics = {"WER (Word Error Rate)": out.wer,
                "CER (Character Error Rate)": jiwer.cer(
                    ref_flat, res_flat, 
                    reference_transform=normalizer, hypothesis_transform=normalizer),
                "MER (Match Error Rate)": out.mer,
                "WIL (Word Information Lost)": out.wil,
                "Substitutions": out.substitutions,
                "Deletions": out.deletions,
                "Insertions": out.insertions,
                "Total Words": out.substitutions + out.deletions + out.hits}
        logger.info(jiwer.visualize_error_counts(out))
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2%}")
            else:
                logger.info(f"{key}: {value}")
        from pprint import pprint
        pprint(out)

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
            env.console; from rich.console import Console; from rich.table import Table  # type: ignore  # noqa: B018, I001
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
        audio_chunk_list = text_chunk_list = lang_chunk_list = saved_safe_start = []
        for res, seg in zip(result.segments, audio_segments):
            if seg.start <= (res.end - res.start) <= seg.end:
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
                # time_batches = ",".join(f"{seg.start},{seg.end}" for seg in audio_segments)
                # batch transcribe expect a dict?
                time_batches = [{"start": seg.start, "end": seg.end} for seg in audio_segments]
                batch_result = self.model.transcribe(audio, language=None, clip_timestamps=time_batches,
                                                    initial_prompt=lyrics, beam_size=self.beamsize, batch_size=16,
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
                    results.append(Segment(words=seg_words))
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
                if seg.start <= (res.end - res.start) <= seg.end:
                    safe_start = max(min(res.start, seg.start), res.start - 1.0)
                    safe_end = min(max(res.end, seg.end), res.end + 1.0)
                    start_sample = int(safe_start * self.sr)
                    end_sample = int(safe_end * self.sr)
                    audio_slice = audio[start_sample:end_sample]
                    assert audio_slice.shape[0] > 0
                    try:
                        align_results = self.model.align(audio_slice, res.text, verbose=None, languange=seg.language or "en") # Auto lang later on.
                        align_results = self.model.refine(audio_slice, result, steps="e", precision=0.5, verbose=None)
                        seg_words = []
                        for new_res in align_results.segments:
                            for word in new_res.words:
                                seg_words.append(WordTiming(
                                    start=float(safe_start + word.start),
                                    end=float(safe_start + word.end),
                                    score=word.score,
                                    word=str(word.text)
                                ))
                        results.append(Segment(words=seg_words))
                    except Exception:
                        logger.exception("!!! Whisper align + refine Error:")
                        raise
            results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
            env.clean()
            return Result(segments=results)