if __name__ == "__main__":
    Wav2Vec2 = type
    QwenASR = type
else:
    from kplus.pipelines.asr.hf.wav2vec2 import Wav2Vec2
    from kplus.pipelines.asr.hf.qwen_asr import QwenASR

from typing import Any, ClassVar

from transformers import AutoConfig


class HFModel:
    _MODEL_MAPPING: ClassVar[dict[str, Any]] = {
        "wav2vec2": Wav2Vec2,
        "qwen3_asr": QwenASR,
    }
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_type = config.model_type
        if (model_class := cls._MODEL_MAPPING.get(model_type)) is None:
            raise ValueError(
                f"Model type '{model_type}' for '{pretrained_model_name_or_path}' is not supported yet! "
                f"Supported types: {list(cls._MODEL_MAPPING.keys())}"
            )
        return model_class(pretrained_model_name_or_path, **kwargs)


if __name__ == "__main__":
    print("main", Wav2Vec2)
    from rich.console import Console
    console = Console()
    console.rule("HF Base Test")
    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    model = HFModel.from_pretrained(model_id)