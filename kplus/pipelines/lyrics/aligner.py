
from kplus import env
from kplus.pipelines.utils import ASRResult, AudioSegment
from kplus.tools import rich

from .audio_aligner import AudioAligner
from .sequence_aligner import SequenceAligner
from .utils import Tokens

__all__ = [
    "LyricAligner"
]

class LyricAligner:
    """ Main For Lyrics Aligner """
    def __init__(self):
        self.sequence_aligner = SequenceAligner()
        self.audio_aligner = AudioAligner()
    
    def asr2ref(self,
        hypothesis: ASRResult,
        reference: str,
        audiosegments: list[AudioSegment]
    ) -> tuple[ASRResult, list[AudioSegment]]:
        ref_tokens, hyp_tokens = (
            self.sequence_aligner(
                Tokens.from_reference(reference),
                Tokens.from_asr(hypothesis.texts)
            )
        )
        datas = self.audio_aligner(ref_tokens, audiosegments)
        new_audiosegments = []
        results = []
        for data in datas:
            assert data.start < data.end
            min_audiosegment = audiosegments[min(data.audio_ids)]
            max_audiosegment = audiosegments[max(data.audio_ids)]
            new_audiosegments.append(AudioSegment(start=min_audiosegment.start, end=max_audiosegment.end))
            results.append(data.to_texttiming())
        if env.verbose:
            for res, aseg in zip(results, new_audiosegments):
                rich.print(res.starth, res.endh, res.text)
                rich.print(aseg.starth, aseg.endh)
                rich.print("="*20)
        return ASRResult(texts=results), new_audiosegments
    