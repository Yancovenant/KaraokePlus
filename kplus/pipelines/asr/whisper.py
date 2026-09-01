from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from kplus import env
from kplus.pipelines.utils import ASRResult, TextTiming, WordTiming
from kplus.tools import filter_known_kwargs, rich
from kplus.tools.audio import Audio

from .base import ASRConfig, ASRMixin

if t.TYPE_CHECKING:
    import numpy as np

    from kplus.pipelines.utils import AudioSegment
    from kplus.tools.audio import AudioNumpy, AudioType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WhisperConfig(ASRConfig):
    max_threads: int = 2
    beam_size: int = 10
    multilingual: bool = True
    patience: float = 2.5
    regroup: bool = False
    compression_ratio_threshold: float = 2.0
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    best_of: float = 5
    vad_filter: bool = False
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    token_step: int = 0 # 0 for max
    extra_models: list[str] = field(default_factory=lambda: ["large-v3", "large-v2", "tiny", "distil-large-v3.5"])
    steps: str = "se"
    whisper_precision: float = 0.02

    # Lang
    language_detection_threshold: float = 0.9
    language_detection_segments: int = 3

    @property
    def initial(self):
        device_index = self.device_map.split(":")[-1]
        device_index = int(device_index) if device_index.isdigit() else 0
        return {
            "device_index": int(device_index),
            "compute_type": str(self.dtype).replace("torch.", ""),
            "num_workers": self.max_threads,
        }


class WhisperASR(ASRMixin):
    """ Whisper class ASR """
    _name = "Whisper"
    
    def __init__(self, modelname: str, **options):
        super().__init__(**options)
        env.stable_ts, env.faster_whisper  # noqa: B018
        import stable_whisper  # type: ignore
        config_params, options = filter_known_kwargs(WhisperConfig, options)
        self.config = WhisperConfig(**config_params)
        options.pop("num_workers", None)
        self.model = stable_whisper.load_faster_whisper(
            modelname, device=env.device.type,
            **self.config.initial,
        )
        self.extra_models = []
        for extra_model in self.config.extra_models:
            self.extra_models.append(
                stable_whisper.load_faster_whisper(
                    extra_model, device=env.device.type,
                    **self.config.initial,
                )
            )

    def detect_language(self,
        audio: AudioType,
        features: np.ndarray | None = None,
        vad_filter: bool = False,
        vad_parameters: dict | None = None,
        language_detection_segments: int | None = None,
        language_detection_threshold: float | None = None
    ) -> str:
        """ Detect Language """
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        lang, prob, all_langs = self.model.detect_language(
            audionp,
            language_detection_threshold=language_detection_threshold or self.config.language_detection_threshold,
            language_detection_segments=language_detection_segments or self.config.language_detection_segments,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters
        )
        logger.debug(f"Detected Language: `{lang}` ({prob})")
        rich.print(rich.Panel(all_langs))
        return lang

    def _transcribe(self, audionp: AudioNumpy, audiosegments: list[AudioSegment], reference: str, prg=None, **kwargs) -> list[TextTiming]:
        results = []
        task = prg.add_task(description="Transcribing...", total=None)
        def progress_callback(seek: float, total: float):
            prg.update(task, completed=seek, total=total)
        # normal transcribe expect str, float() timestamp
        time_batches = ",".join(f"{seg.start},{seg.end}" for seg in audiosegments)
        # batch transcribe expect a dict?
        # time_batches = [{"start": seg.start, "end": seg.end} for seg in audio_segments]
        batch_result = self.model.transcribe(
            audionp, language=None,
            initial_prompt=reference,
            multilingual=kwargs.pop("multilingual", self.config.multilingual),
            beam_size=kwargs.pop("beam_size", self.config.beam_size),
            patience=kwargs.pop("patience", self.config.patience),
            regroup=kwargs.pop("regroup", self.config.regroup),
            language_detection_threshold=kwargs.pop("language_detection_threshold", self.config.language_detection_threshold),
            language_detection_segments=kwargs.pop("language_detection_segments", self.config.language_detection_segments),
            compression_ratio_threshold=kwargs.pop("compression_ratio_threshold", self.config.compression_ratio_threshold),
            log_prob_threshold=kwargs.pop("log_prob_threshold", self.config.log_prob_threshold),
            no_speech_threshold=kwargs.pop("no_speech_threshold", self.config.no_speech_threshold),
            best_of=kwargs.pop("best_of", self.config.best_of),
            vad_filter=kwargs.pop("vad_filter", self.config.vad_filter),
            temperature=kwargs.pop("temperature", self.config.temperature),
            clip_timestamps=time_batches,
            #batch_size=1, # need to be None
            repetition_penalty=1.2,
            condition_on_previous_text=False,
            progress_callback=progress_callback,
            verbose=kwargs.pop("verbose", None),
            #**kwargs
        )
        for res in batch_result:
            words = []
            for w in res.words:
                words.append(
                    WordTiming(
                        start=float(w.start),
                        end=float(w.end),
                        score=float(w.probability),
                        word=str(w.word)
                    )
                )
            results.append(
                TextTiming(
                    words=words,
                    language=batch_result.language
                )
            )
        return results

    def _align(self, audionp: AudioNumpy, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], prg=None, **kwargs) -> list[TextTiming]:
        results = []
        task = prg.add_task(description="Aligning...", total=None)
        def progress_callback(seek: float, total: float):
            prg.update(task, completed=seek, total=total)
        for hyp, seg in zip(transcriptions.texts, audiosegments):
            if not (seg.end < hyp.start or seg.start > hyp.end):
                safe_start = max(0, max(min(hyp.start, seg.start), hyp.start - 1.0) - 0.5)
                safe_end = min(len(audionp), min(max(hyp.end, seg.end), hyp.end + 1.0) + 0.5)
                audio_chunk = Audio.slicenp(audionp, safe_start, safe_end, self.sr)
                assert len(audio_chunk) > 0
                align_result = self.model.align(
                    audio_chunk, hyp.latin,
                    language=hyp.language,
                    token_step=kwargs.pop("token_step", self.config.token_step),
                    regroup=kwargs.pop("regroup", self.config.regroup),
                    vad=kwargs.pop("vad_filter", self.config.vad_filter),
                    extra_models=self.extra_models,
                    progress_callback=progress_callback,
                    verbose=kwargs.pop("verbose", None),
                )
                align_result = self.model.refine(
                    audio_chunk, align_result,
                    steps=kwargs.pop("steps", self.config.steps),
                    precision=kwargs.pop("whisper_precision", self.config.whisper_precision),
                    verbose=kwargs.pop("verbose", None),
                )
                words = []
                for res in align_result.segments:
                    if len(res.words) != len(hyp.words):
                        res = self._fix_duplicate(res, hyp)
                    assert len(res.words) == len(hyp.words), f"Word missmatch {res.text} | {len(res.words)} | {len(hyp.words)}"
                    for w in res.words:
                        words.append(WordTiming(
                            start=float(round(safe_start + w.start, 2)), end=float(round(safe_start + w.end,3)),
                            score=float(round(w.probability, 3)), word=str(w.word.strip()),
                        ))
                results.append(TextTiming(words=words, language=align_result[0].language))
        return results