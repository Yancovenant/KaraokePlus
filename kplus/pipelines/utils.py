from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import kplus

if TYPE_CHECKING:
    import numpy as np, torch # type: ignore  # noqa: I001
    AudioType : TypeAlias = "torch.Tensor | np.ndarray | str"


def load_audio(audio_path: str, sr: float, channels: int) -> torch.Tensor:
    kplus.env.demucs  # noqa: B018
    from demucs.audio import AudioFile  # type: ignore
    return AudioFile(str(audio_path)).read(
        streams=0, samplerate=sr, channels=channels
    )

def convert_audio(audio: torch.Tensor, fromsr: float, tosr: float, channels=int) -> torch.Tensor:
    kplus.env.demucs  # noqa: B018
    from demucs.audio import convert_audio as julius_resampler  # type: ignore
    return julius_resampler(audio, fromsr, tosr, channels)
