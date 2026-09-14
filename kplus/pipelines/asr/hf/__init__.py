#from __future__ import annotations
if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path
    target_dir = Path("~/storage/shared/download/JAZPIPER_AUTOMATION/KaraokePlus").expanduser().resolve()
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))
    import kplus.init
    
from kplus.pipelines.asr.hf.wav2vec2 import Wav2Vec2

class HFModel:
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        if "facebook" in pretrained_model_name_or_path:
            return Wav2Vec2(pretrained_model_name_or_path, **kwargs)

if __name__ == "__main__":
    print("main", Wav2Vec2)