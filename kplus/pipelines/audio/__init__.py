

from .detection import DetectionResult, AudioExtractor, AudioSegment

__all__ = [
    "detect_audio_activity",
    "AudioSegment",
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