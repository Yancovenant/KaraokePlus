from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
import kplus

if TYPE_CHECKING:
    import torch
    import numpy as np
    AudioType : TypeAlias = "torch.Tensor | np.ndarray | str"


def load_audio(audio_path: str, sr: float, channels: int) -> torch.Tensor:
    kplus.env.demucs
    from demucs.audio import AudioFile
    return AudioFile(str(audio_path)).read(
        streams=0, samplerate=sr, channels=channels
    )

def convert_audio(audio: torch.Tensor, fromsr: float, tosr: float, channels=int) -> torch.Tensor:
    kplus.env.demucs
    from demucs.audio import convert_audio as julius_resampler
    return julius_resampler(audio, fromsr, tosr, channels)