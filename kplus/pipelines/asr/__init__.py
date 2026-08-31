from __future__ import annotations

import typing as t

from kplus import env

env.torch
import torcb

from kplus.pipelines.utils import ASRResult

from .base import MMS_FA, ASRMixin
from .qwen import QwenASR
from .whisper import WhisperASR

if t.TYPE_CHECKING:
    from kplus.pipelines.utils import AudioSegment
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
        whisper_modelname = options.pop("whisper", None)
        qwen_modelname = options.pop("qwen", None)
        is_mms = options.pop("mms_fa", None)
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
    result = transcriber.transcribe(audio, audiosegments, reference, **options)
    del transcriber.model, transcriber
    env.clean()
    return result

def align(audio: AudioType, audiosegments: list[AudioSegment], reference: str, **options):
    """ Single Align """
    with torch.inference_mode():
        aligner = BaseASR.from_model(**options)
        result = aligner.align(audio, audiosegments, reference, **options)
    del aligner.model, aligner
    env.clean()
    return result

def multi_align(audio: AudioType, audiosegments: list[AudioSegment], reference: str, **options):
    """ Multiple Model Alignment """
    whisper_align = align(audio, audiosegments, reference, whisper="large-v3", **options)
    qwen_align = align(audio, audiosegments, reference, qwen="Qwen/Qwen3-ASR-1.7B", **options)
    mms_align = align(audio, audiosegments, reference, mms_fa=True, **options)
    return whisper_align, qwen_align, mms_align
