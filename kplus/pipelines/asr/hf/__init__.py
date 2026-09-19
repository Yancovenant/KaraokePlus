from typing import Any, ClassVar

from .qwen_asr import QwenASR
from .wav2vec2 import Wav2Vec2


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
        from transformers import AutoConfig
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
    # model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    model_id = "facebook/mms-1b-all"
    model = HFModel.from_pretrained(model_id)
    from rich import inspect
    inspect(model)