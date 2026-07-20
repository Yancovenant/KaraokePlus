from __future__ import annotations

from typing import TYPE_CHECKING, List

from kplus.environment import env

if TYPE_CHECKING:
    import torch
    import numpy as np
    from .aad import AudioSegment

class Aligner:
    def __init__(self):
        pass
    
    def get_audio_segments(self, y: np.ndarray) -> List[AudioSegment]:
        pass

    def main(self, audio: torch.Tensor, sr: float, lyrics: str):
        env.demucs
        from demucs.audio import convert_audio
        audio = convert_audio(audio, sr, self.sr, channels=1)
        audio_np = audio.detach().cpu().numpy().squeeze().copy()
        audio_segments = self.get_audio_segments(audio_np)
        transcripts = self.get_transcript(audio_np, audio_segments, lyrics)
        results = self.adjust_by_lyrics(transcripts, lyrics, audio_segments, 0.0)
        refine_results = self.ctc_align(audio, results, audio_segments)
        refine_results = self.refine_segments_with_dsp(refine_results, audio_np, self.sr)
        refine_results.to_lyrics_segment()
        del audio, audio_np
        env.clean()
        return refine_results