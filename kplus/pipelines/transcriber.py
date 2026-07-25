from __future__ import annotations

import difflib
import logging
import string
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .aad import AudioSegment

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType


logger = logging.getLogger(__name__)

@dataclass
class WordTiming:
    start: float
    end: float
    score: float
    word: str

@dataclass(slots=True)
class Segment:
    words: list[WordTiming]

    @property
    def text(self) -> str:
        return " ".join([w.word for w in self.words])

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

@dataclass(slots=True)
class Result:
    segments: list[Segment]

    def to_lyrics_segment(self):
        new_segments = []
        all_words = [w for segs in self.segments for w in segs.words]
        for idx, group in groupby(all_words, key=lambda x: x.line_idx):
            words = list(group)
            new_segments.append(Segment(words=words))
        self.segments = new_segments
        return self


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
        env.numpy, env.torch  # noqa: B018
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
        

    def get_lyrics_timestamp(self, transcripts: Result, lyrics: str, audio_segments: list[AudioSegment], max_bleed_seconds: float = 0.5) -> tuple[Result, list[AudioSegment]]:
        """ Synchronizes raw lyrics with audio transcripts and segments, 
            creating a temporally aligned result.
        """
        env.sequence_align  # noqa: B018
        from sequence_align.pairwise import needleman_wunsch_with_scores  # type: ignore
        _PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)
        def clean_word(text: str) -> str: return text.translate(_PUNCTUATION_TRANSLATOR).lower().strip()
        if not lyrics or not transcripts.segments or not audio_segments: return Result(segments=[]), []

        lyric_lines = [line.strip() for line in lyrics.split("\n") if line.strip() and not line.startswith('[')]
        lyric_tokens = [
            (token:=WordTiming(word=word, start=None, end=None, score=None),
             setattr(token, "clean", clean_word(word)),
             setattr(token, "line_idx", line_idx))[0]
            for line_idx, line in enumerate(lyric_lines)
            for word in line.split()
        ]
        transcript_tokens = [
            (token:=WordTiming(word=word.word, start=word.start, end=word.end, score=word.score),
             setattr(token, "clean", clean_word(word.word)))[0]
            for segment in transcripts.segments
            for word in segment.words if word.score > 0.35
        ]
        lyric_clean = [t.clean for t in lyric_tokens]
        transcript_clean = [t.clean for t in transcript_tokens]
        def fuzzy_score(a, b):
            if a == b: return 2.0  # Perfect match
            if difflib.SequenceMatcher(None, a, b).ratio() > 0.6: return 1.0  # High similarity (slight mishearings)
            return -3.0
        aligned_lyric, aligned_transcript = needleman_wunsch_with_scores(
            lyric_clean, transcript_clean,
            gap="-", score_fn=fuzzy_score, indel_score=-1.0
        )
        lyric_idx, transcript_idx = 0, 0
        for lyric_word, transcript_word in zip(aligned_lyric, aligned_transcript):
            if lyric_word == "-": transcript_idx += 1; continue
            if transcript_word == "-": lyric_idx += 1; continue
            if fuzzy_score(lyric_word, transcript_word) > 0:
                lyric_token = lyric_tokens[lyric_idx]
                transcript_token = transcript_tokens[transcript_idx]
                lyric_token.start = transcript_token.start
                lyric_token.end = transcript_token.end
                lyric_token.score = transcript_token.score
                logger.debug(f"{'':<4}Matched: {lyric_token.word} to {transcript_token.word}")
            lyric_idx += 1
            transcript_idx += 1
        lines_map: dict[int, list[WordTiming]] = defaultdict(list)
        for token in lyric_tokens: lines_map[token.line_idx].append(token)

        seg_midpoints = [(s.start + s.end) / 2 for s in audio_segments]
        seg_index_map = {seg: i for i, seg in enumerate(audio_segments)}
        line_to_segments: dict[int, list[AudioSegment]] = {}
        for idx in range(len(lyric_lines)):
            line_words = lines_map[idx]
            anchors = [w for w in line_words if w.start is not None]
            if anchors:
                matched_segs = []
                for w in anchors:
                    mid = (w.start + w.end) / 2
                    min_dist = float('inf')
                    closest_seg = audio_segments[0]
                    for i, seg in enumerate(audio_segments):
                        if seg.start <= mid <= seg.end: closest_seg = seg; break
                        dist = abs(mid - seg_midpoints[i])
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
        super_phrases: list[tuple[list[WordTiming], AudioSegment]] = []
        sorted_indices = sorted(lines_map.keys())
        if sorted_indices:
            current_words = lines_map[sorted_indices[0]]
            current_segs = set(line_to_segments[sorted_indices[0]])
            for idx in sorted_indices[1:]:
                next_segs = set(line_to_segments[idx])
                next_words = lines_map[idx]
                if current_segs.isdisjoint(next_segs):
                    min_start = min(s.start for s in current_segs)
                    max_end = max(s.end for s in current_segs)
                    super_phrases.append((current_words, AudioSegment(start=min_start, end=max_end)))
                    
                    current_words = next_words
                    current_segs = next_segs
                else:
                    logger.debug(f"overlap! {current_segs & next_segs}")
                    current_words.extend(next_words)
                    current_segs.update(next_segs)
            min_start = min(s.start for s in current_segs)
            max_end = max(s.end for s in current_segs)
            super_phrases.append((current_words, AudioSegment(start=min_start, end=max_end)))
        final_segments = []
        final_audio_segments = []
        for words, segment in super_phrases:
            allowed_start = segment.start - max_bleed_seconds
            allowed_end = segment.end + max_bleed_seconds
            n = len(words)
            i = 0
            while i < n:
                if words[i].start is None:
                    logger.debug(f"{'':<2} Dropped: {words[i].word}")
                    block_start = i
                    while i < n and words[i].start is None:
                        i += 1
                    block_end = i
                    prev_end = allowed_start
                    if block_start > 0:
                        logger.debug(f"{'':<4} Found leading end: {words[block_start - 1].word}-> {words[block_start - 1].end}")
                        prev_end = words[block_start - 1].end

                    next_start = allowed_end
                    if block_end < n:
                        logger.debug(f"{'':<4} Found leading end: {words[block_end].word}-> {words[block_end].end}")
                        next_start = words[block_end].start

                    gap = next_start - prev_end
                    time_per_word = gap / (block_end - block_start + 1)
                    curr_time = prev_end
                    for j in range(block_start, block_end):
                        logger.debug(f"{'':<8} Interpolate: [None - None] to [{curr_time:.2f}s - {curr_time + time_per_word:.2f}s]")
                        words[j].start = curr_time
                        words[j].end = curr_time + time_per_word
                        curr_time += time_per_word
                else:
                    i += 1
            if self.verbose:
                self._plot_jiwer(lyrics, transcripts)
            final_segments.append(Segment(words=words))
            final_audio_segments.append(segment)
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

    def _apply_model(self, audio_segments: list[AudioSegment], audio: list[tuple[np.ndarray, int]], context: list[str] | None):
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
                        results.append(Segment(words=seg_words))
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
        audio_chunk_list = []
        text_chunk_list = []
        for res, seg in zip(result.segments, audio_segments):
            if seg.start <= (res.end - res.start) <= seg.end:
                safe_start = min(res.start, seg.start)
                safe_end = max(res.end, seg.end)
                start_sample = int(safe_start * self.sr)
                end_sample = int(safe_end * self.sr)
                audio_slice = audio[:, start_sample:end_sample]
                assert audio_slice.shape[0] > 0
                audio_chunk_list.append((audio_slice, self.sr))
                text_chunk_list.append(res.text)
        results = self._apply_model(audio_segments=audio_segments, audio=audio_chunk_list, context=text_chunk_list)
        return Result(segments=results)


class WhisperTranscribe(TranscriberMixin):
    def __init__(self, options):
        super().__init__(options)
        env.stable_ts, env.faster_whisper  # noqa: B018
        import stable_whisper  # type: ignore
        self.max_threads = max(1, options.max_threads)
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
                                                    initial_prompt=lyrics, beam_size=5, batch_size=16,
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
     
