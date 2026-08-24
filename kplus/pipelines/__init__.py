#from .songdownloader import *
from .audioseparation import SeparatorMixin
from .visualizer import VisualizeWaveform
from .aligner import *
from .aad import *
from .transcriber import *
from .refiner import TimestampRefiner


__all__ = [
    "separate"
]

def separate(**options):
    import subprocess
    from pathlib import Path
    from kplus.pipelines import SeparatorMixin
    filepath = Path(info.filename)
    audio_file_path = f"{filepath.stem}.wav"
    options = SimpleNamespace(modelname="demucs", overlap=0.75, segment=200, shifts=1)
    sep_class = SeparatorMixin.get_model(options)
    subprocess.run(["ffmpeg", "-y", "-i", str(filepath), "-vn", "-ar", str(sep_class.sr), "-ac", str(sep_class.ac), str(audio_file_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sep_info = sep_class.separate(audio_file_path)
    inspect(sep_info)
    vocs_path = sep_info.vocs_path
    inst_path = sep_info.inst_path
    try:
        del sep_class.model, sep_class
    except:
        pass
    env.clean()
    return vocs_path, inst_path, sep_info
