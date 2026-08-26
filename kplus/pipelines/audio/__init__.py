from __future__ import annotations

import typing as t

from .detection import AudioExtractor, AudioSegment, DetectionResult

if t.TYPE_CHECKING:
    from kplus.tools.audio import AudioType

__all__ = [
    "AudioSegment",
    "detect_audio_activity",
]

def detect_audio_activity(
    audio : AudioType,
    sr: int,
    *,
    precision_ms: int = 10,
    signal_overlap: float = 0.75,
    use_filter: bool = True,
    **kwargs
) -> DetectionResult:
    extractor = AudioExtractor(precision_ms, signal_overlap, use_filter, **kwargs)
    return extractor.detect_all(audio, sr)
