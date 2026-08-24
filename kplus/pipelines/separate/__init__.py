import subprocess
import typing as t
from pathlib import Path

from kplus import env
from kplus.pipelines.download import DownloadResult

from .base import SeparationResult
from .demucs import DemucsSeparator

__all__ = [
    "SeparationResult",
    "separate_song",
]

class BaseSeparator:
    modelclass: t.ClassVar = {"demucs": DemucsSeparator}
    
    @classmethod
    def from_options(self, **options):
        demucs_modelname = options.pop("demucs")
        uvr_modelname = options.pop("uvr")
        if demucs_modelname and uvr_modelname:
            raise ValueError("Cannot use Demucs model and UVR model at the same time: %s - %s", demucs_modelname, uvr_modelname)
        modelclass = self.modelclass["demucs" if demucs_modelname else "uvr"]
        modelname = demucs_modelname if demucs_modelname else uvr_modelname
        return modelclass(modelname, **options)


def separate_song(info: DownloadResult, **options) -> SeparationResult:
    inputpath = Path(info.filepath).expanduser().resolve()
    mixpath = inputpath.stem + ".wav"
    separator = BaseSeparator.from_options(**options)
    # Normalize and Get Audio
    subprocess.run(["ffmpeg", "-y", "-i", str(inputpath), "-vn", "-ar", str(separator.sr), "-ac", str(separator.ac), str(mixpath)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = separator.separate(mixpath)
    del separator.model, separator
    env.clean()
    return result