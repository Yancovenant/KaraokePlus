from __future__ import annotations

import typing as t

from .base import MMS_FA, ASRMixin, QwenASR
from .utils import ASRResult
from .whisper import WhisperASR

if t.TYPE_CHECKING:
    from kplus.pipelines.audio import AudioSegment
    from kplus.tools.audio import AudioType

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
    def from_model(cls, **options) -> ASRMixin:
        whisper_modelname = options.pop("whisper")
        qwen_modelname = options.pop("qwen")
        is_mms = options.pop("mms_fa")
        error_text = f"Cannot use multiple model at the same time: {whisper_modelname} - {qwen_modelname} - {is_mms}"
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
