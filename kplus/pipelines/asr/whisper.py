from __future__ import annotations

import typing as t
from dataclasses import asdict, dataclass

from kplus import env
from kplus.tools import filter_known_kwargs

from .base import ASRConfig, ASRMixin
from .utils import TextTiming, WordTiming

if t.TYPE_CHECKING:
    from kplus.pipelines.audio import AudioSegment
    from kplus.tools.audio import AudioNumpy


@dataclass(slots=True)
class WhisperConfig(ASRConfig):
    max_threads: int = 2
    beam_size: int = 10
    multilingual: bool = True
    patience: float = 2.5
    regroup: bool = False
    language_detection_threshold: float = 0.9
    compression_ratio_threshold: float = 2.0
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    best_of: float = 5
    vad_filter: bool = False
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

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
    