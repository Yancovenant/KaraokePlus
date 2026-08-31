
from kplus.pipelines.utils import ASRResult, AudioSegment

from .aligner import LyricAligner
from .audio_aligner import AudioAligner
from .sequence_aligner import SequenceAligner

__all__ = [
    "AudioAligner",
    "LyricAligner",
    "SequenceAligner",
    "align2ref",
]

def align2ref(hypothesis: ASRResult, reference: str, audiosegments: list[AudioSegment]):
    return LyricAligner().asr2ref(hypothesis, reference, audiosegments)