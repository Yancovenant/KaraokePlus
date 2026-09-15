from __future__ import annotations

import typing as t

import torch

from kplus import env
from kplus.pipelines.utils import ASRResult
from kplus.tools import filter_known_kwargs

from .base import MMS_FA
from .hf import HFModel
from .qwen import QwenASR
from .whisper import WhisperASR

if t.TYPE_CHECKING:
    from kplus.pipelines.utils import AudioSegment
    from kplus.tools.audio import AudioType

    from .hf.mixin import ASRMixin


__all__ = [
    "align",
    "detect_language",
    "transcribe",
]

class BaseASR:
    """ Base Class for asr model """
    modelclass: t.ClassVar = {
        "whisper": WhisperASR,
        "qwen": QwenASR,
        "mms_fa": MMS_FA
    }
    @classmethod
    def from_model(cls, **options):
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


def detect_language(audio: AudioType, **options) -> str:
    """ Detect language. """
    transcriber: WhisperASR = BaseASR.from_model(whisper="large-v3", extra_models=[])
    detection_params, options = filter_known_kwargs(transcriber.detect_language, options)
    lang = transcriber.detect_language(audio, **detection_params)
    del transcriber.model, transcriber
    env.clean()
    return lang


def transcribe(audio: AudioType, audiosegments: list[AudioSegment], reference:str, *, languages: str | list[str] | None = None, **options) -> ASRResult:
    """ Transcribe given audio file. """
    model_name_or_path = options.pop("transcribe_model_name_or_path", "Qwen/Qwen3-ASR-1.7B-hf")
    return_timestamps = options.pop("transcribe_return_timestamps", True)
    transcriber: ASRMixin = HFModel.from_pretrained(model_name_or_path, **options)
    result = transcriber.transcribe(
        audio=audio,
        audiosegments=audiosegments,
        contexts=reference,
        languages=languages,
        return_timestamps=return_timestamps
    )
    try:
        del transcriber.model, transcriber.processor, transcriber
    except:  # noqa: E722, S110
        pass
    env.clean()
    return result


def align(audio: AudioType, transcriptions: ASRResult, audiosegments: list[AudioSegment], **options):
    """ Single Align """
    with torch.inference_mode():
        model_name_or_path = options.pop("align_model_name_or_path", "facebook/mms-1b-all")
        aligner: ASRMixin = HFModel.from_pretrained(model_name_or_path, **options)
        result = aligner.align(
            audio=audio,
            audiosegments=audiosegments,
            hypothesis=transcriptions.texts,
            languages=None,
        )
    try:
        del aligner.model, aligner.processor, aligner
    except:  # noqa: E722, S110
            pass
    env.clean()
    return result
