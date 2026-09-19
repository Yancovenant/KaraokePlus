from __future__ import annotations

from kplus.pipelines.utils import ASRResult, AudioSegment

from .aligner import LyricAligner
from .utils import LyricAlignError

__all__ = [
    "LyricAlignError",
    "align2ref",
]


def align2ref(
    hypothesis: ASRResult,
    reference: str,
    audiosegments: list[AudioSegment],
    *,
    raise_if_not_reliable: bool = True,
    **kwargs
) -> ASRResult:
    return LyricAligner(
        raise_if_not_reliable=raise_if_not_reliable
    ).asr2ref(hypothesis, reference, audiosegments)
