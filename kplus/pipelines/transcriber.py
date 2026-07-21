from __future__ import annotations
import concurrent.futures
import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from itertools import groupby

from kplus.environment import env
from .aad import AudioSegment
from kplus.tools.progress import MainProgress

if TYPE_CHECKING:
    import numpy as np
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
    segments: List[Segment]

    def to_lyrics_segment(self):
        new_segments = []
        all_words = [w for segs in self.segments for w in segs.words]
        for idx, group in groupby(all_words, key=lambda x: x.line_idx):
            words = list(group)
            new_segments.append(Segment(words=words))
        self.segments = new_segments
        return self


class Transcriber:
    def __init__(self, max_threads: int = 2, model_name: str = "large-v3", use_cliptimestamp: bool = False):
        self.max_threads = max(1, max_threads) # min 1
        # We expect the samplerate to be 16000
        env.stable_ts, env.faster_whisper
        import stable_whisper
        compute_type = "float16" if env.device.type == "cuda" else "float32"
        self.model = stable_whisper.load_faster_whisper(model_name, device=env.device.type, compute_type=compute_type, num_workers=max_threads)
        self.sr = self.model.feature_extractor.sampling_rate
        self.use_cliptimestamp = use_cliptimestamp

    def _process_chunk(self, audio: np.ndarray, segment, lyrics):
        start_sample =int(segment.start * self.sr)
        end_sample = int(segment.end * self.sr)
        chunk = audio[start_sample:end_sample].squeeze()
        lang, _, _ = self.model.detect_language(chunk, language_detection_threshold=0.9, language_detection_segments=2)
        result = self.model.transcribe(chunk, language=lang, initial_prompt=lyrics, vad=False, beam_size=5, patience=2,
                                       condition_on_previous_text=False, repetition_penalty=1.2)
        chunk_segments = []
        for res in result:
            chunk_segments.append(
                Segment(words=list(WordTiming(
                    start=float(w.start + segment.start),
                    end=float(w.end + segment.start),
                    score=float(w.probability), word=str(w.word)) for w in res.words))
            )
        del chunk, lang, result
        return chunk_segments
    
    def transcribe(self, audio: AudioType, audio_segments, lyrics: str, sr: Optional[float]):
        env.numpy, env.torch
        import numpy as np, torch
        if not isinstance(audio, (np.ndarray, torch.Tensor)):
            from .utils import load_audio
            audio = load_audio(audio, self.sr, 1)
        if isinstance(audio, torch.Tensor):
            from .utils import convert_audio
            audio = convert_audio(audio, sr, self.sr, 1)
            audio = audio.detach().cpu().numpy()
        audio = audio.squeeze()
        results = []
        if not audio_segments:
            duration = len(audio) / self.sr
            audio_segments = [AudioSegment(start=0.0, end=duration)]
        with MainProgress(total = len(audio_segments), desc="Starting Transcriptions...", unit="chunk") as main_bar:
            try:
                if not self.use_cliptimestamp:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                        future_to_seg = {executor.submit(self._process_chunk, audio, seg, lyrics): seg for seg in audio_segments}
                        for future in concurrent.futures.as_completed(future_to_seg):
                            try:
                                chunk_result = future.result()
                                results.extend(chunk_result)
                                main_bar.update(1)
                            except Exception as e:
                                logger.error(f"!!! Transcriber failed to process chunk: {e}", exc_info=True)
                else:
                    time_batches = [{"start": float(seg.start), "end": float(seg.end)} for seg in audio_segments]
                    batch_result = self.model.transcribe_string_batches(
                        audio,
                        batches=time_batches,
                        batch_size=16,          # <-- THE GPU ACCELERATOR
                        language=None,          # Auto-detects language once
                        initial_prompt=lyrics,
                        beam_size=5,
                        repetition_penalty=1.2,
                        condition_on_previous_text=False
                    )
            except Exception as err:
                logger.error(f"!!! Concurrent Error: {err}", exc_info=True)
                raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        logger.debug(">> Whisper Transcription:")
        for res in results:
            logger.debug(f"{'':<2}Segment: {res.start:.2f}s - {res.end:.2f}s (Duration: {res.end-res.start:.2f}s)")
            for w in res.words:
                logger.debug(f"{'':<4}WordTiming: ({w.score:.2f}) {w.start:.2f}s to {w.end:.2f}s {w.word}")
        env.clean()
        return Result(segments=results)
