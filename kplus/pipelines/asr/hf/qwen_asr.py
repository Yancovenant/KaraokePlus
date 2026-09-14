from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from .mixin import ASRMixin


class QwenASR(ASRMixin):
    """ Alibaba Qwen ASR Model Class. """
    force_align_model_id_or_path: str = ""
    
    def _load_model(self, model_name_or_path, **kwargs) -> None:
        self.model = AutoModelForMultimodalLM(model_name_or_path, **kwargs)
        self.processor = AutoProcessor(model_name_or_path, **kwargs)
        self.force_aligner = AutoModelForTokenClassification(self.force_align_model_id_or_path, **kwargs)
        self.force_aligner_processor = AutoProcessor(self.force_align_model_id_or_path, **kwargs)
        