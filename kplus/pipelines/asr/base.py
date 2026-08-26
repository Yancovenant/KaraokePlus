from __future__ import annotations

from kplus.tools.audio import Audio, AudioType, AudioNumpy
# Need to be below
import torch
import typing as t
import logging

from dataclasses import field, dataclass, asdict

from kplus import env
from kplus.tools import rich

from .utils import get_default_dtype, TextTiming, WordTiming, ASRResult

if t.TYPE_CHECKING:
    from kplus.pipelines.audio import AudioSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ASRConfig:
    dtype: torch.dtype = field(default_factory=get_default_dtype)
    device_map: str = (
        "cuda:"
        "1" if torch.cuda.device_count() > 1 else "0"
    ) if env.device.type == "cuda" else env.device.type
    

class ASRMixin:
    """ ASR Mixin """
    def __init__(self, **options):
        if "dtype" in options:
            dtype = options.pop("dtype").lower()
            if dtype != "auto":
                try:
                    options["dtype"] = getattr(torch, dtype)
                except AttributeError:
                    raise ValueError("dtype not recognize %s", dtype)
        self.sr = 16000 # Used for all model

    def _transcribe(self, audio: AudioNumpy, audiosegments: list[AudioSegment], reference: str, prg=None, **kwargs) -> list[TextTiming]:
        raise NotImplementedError()

    def transcribe(self, audio: AudioType, audiosegments: list[AudioSegment], reference: str, **kwargs) -> ASRResult:
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        if not audiosegments:
            duration = len(audionp) / self.sr
            audiosegments = [AudioSegment(start=0.0, end=duration)]
        try:
            with rich.make_progress(is_download=False) as prg:
                prg.add_task(description=f"{self._name} Starting ASR...", total=None)
                results = self._transcribe(audionp, audiosegments, reference, prg, **kwargs)
        except Exception as err:
            logger.exception(f"Error while doing transcriptions, error: {err}")
            raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        return ASRResult(texts=results)


@dataclass(slots=True)
class QwenConfig(ASRConfig):
    max_inference_batch_size: int = -1 # -1 for infinte
    max_new_tokens: int = 8192 #
    num_beams: int = 10

class QwenASR(ASRMixin):
    """ Qwen class ASR """
    _name = "Qwen"
    
    def __init__(self, modelname: str, **options):
        super().__init__(self, **options)
        env.torchvision, env.qwen_asr  # noqa: B018
        from qwen_asr import Qwen3ASRModel  # type: ignore
        self.model = Qwen3ASRModel.from_pretrained(
            #"Qwen/Qwen3-ASR-1.7B",
            modelname,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
            attn_implementation="sdpa",
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            **options,
            **asdict(self.config),
            forced_aligner_kwargs={
                "dtype": self.config.dtype,
                "device_map": self.config.device_map,
                "attn_implementation": "sdpa",
                **options,
                **asdict(self.config),
            }
        )
        self.config = QwenConfig(**options)

    def _transcribe(self, audionp: AudioNumpy, audiosegments: list[AudioSegment], reference: str, prg=None, **kwargs) -> list[TextTiming]:
        logger.debug(f"Running Qwen ASR model with {len(audiosegments)} segments and config: {self.sr}")
        audio_chunk_list, results = [], []
        for seg in audiosegments:
            start, end = int(seg.start * self.sr), int(seg.end * self.sr)
            audio_chunk_list.append((audionp[start:end], self.sr))
        logger.debug(f"Prepared {len(audio_chunk_list)} audio chunks for Qwen ASR model")
        batch_result = self.model.transcribe(audio=audio_chunk_list, context=None, return_time_stamps=True, **kwargs)
        logger.debug(f"Qwen ASR model returned {len(batch_result)} segments")
        for seg, aseg in zip(batch_result, audiosegments):
            #prg.update(1)
            if seg.time_stamps is not None:
                words = []
                for word in seg.time_stamps:
                    words.append(
                        WordTiming(
                            start=float(word.start_time + aseg.start),
                            end=float(word.end_time + aseg.start),
                            score=1.0,
                            word=str(word.text)
                        )
                    )
                results.append(
                    TextTiming(
                        words=words,
                        language=seg.language
                    )
                )
        return results

class MMS_FA(ASRMixin):
    """ mms_fa facebook wav2vec2 aligner """