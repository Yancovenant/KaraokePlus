from __future__ import annotations

import typing as t

from kplus.pipelines.utils import ASRResult, AudioSegment

from .aligner import LyricAligner
from .audio_aligner import AudioAligner
from .refiner import Refiner
from .sequence_aligner import SequenceAligner

if t.TYPE_CHECKING:
    from kplus.tools.audio import AudioType

__all__ = [
    "AudioAligner",
    "LyricAligner",
    "SequenceAligner",
    "align2ref",
    "refine",
]

def align2ref(hypothesis: ASRResult, reference: str, audiosegments: list[AudioSegment], *, raise_if_not_reliable: bool = True, **kwargs):
    return LyricAligner(raise_if_not_reliable=raise_if_not_reliable).asr2ref(hypothesis, reference, audiosegments)

def refine(audio: AudioType, original: ASRResult, *ai_res, audiosegments: list[AudioSegment]) -> ASRResult:
    return Refiner()(audio, original, *ai_res, audiosegments=audiosegments)