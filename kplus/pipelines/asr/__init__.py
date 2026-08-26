
import typing as t

from .base import ASRMixin, QwenASR, MMS_FA
from .whisper import WhisperASR
from .utils import ASRResult

if t.TYPE_CHECKING:
    from kplus.tools.audio import AudioType
    from kplus.pipelines.audio import AudioSegment

__all__ = [
    "transcribe"
]

class BaseASR:
    """ Base Class for asr model """
    modelclass: t.ClassVar = {
        "whisper": WhisperASR,
        "qwen": QwenASR,
        "mms_fa": MMS_FA
    }
    @classmethod
    def from_model(cls, **options) -> "ASRMixin":
        whisper_modelname = options.pop("whisper")
        qwen_modelname = options.pop("qwen")
        is_mms = options.pop("mms_fa")
        error_text = f"Cannot use multiple model at the same time: {whisper_modelname} - {qwen_modelname} - {mms_fa}"
        if (
            (whisper_modelname and qwen_modelname)
            or (whisper_modelname and is_mms)
            or (qwen_modelname and is_mms)
        ):
            raise ValueError(error_text)
        modelclass = cls.modelclass[
            "whisper" if whisper_modelname
            else ("qwen" if qwen_modelname
            else "mms_fa")
        ]
        modelname = (
            whisper_modelname if whisper_modelname
            else (qwen_modelname if qwen_modelname
            else None)
        )
        return modelclass(modelname, **options)

def transcribe(audio: AudioType, audiosegments: list[AudioSegment], reference:str, **options) -> ASRResult:
    """ Transcribe given audio file """
    transcriber = BaseASR.from_model(**options)
    return transcriber.transcribe(audio, audiosegments, reference, **options)
