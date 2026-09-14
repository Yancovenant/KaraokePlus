import logging

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from kplus import env
from kplus.pipelines.utils import TextTiming, WordTiming
from kplus.tools.audio import AudioInput, AudioNumpy, IndexAudioInput

from ..kwargs_utils import ASRKwargs, merge_kwargs
from .mixin import ASRMixin

logger = logging.getLogger(__name__)


class QwenASRKwargs(ASRKwargs):
    _defaults = {  # noqa: RUF012
        "processor_kwargs": {
            "return_tensors": "pt",
            "padding": True,
        },
        "tokenizer_kwargs": {
            "return_tensors": "pt",
            "padding": True,
        }
    }

class QwenASR(ASRMixin):
    """ Alibaba Qwen ASR Model Class. """
    force_align_model_id_or_path: str = ""
    
    def _load_model(self, model_name_or_path, **kwargs) -> None:
        self.model = AutoModelForMultimodalLM(model_name_or_path, **kwargs).to(env.device).eval()
        self.processor = AutoProcessor(model_name_or_path, **kwargs)
        self.force_aligner = AutoModelForTokenClassification(self.force_align_model_id_or_path, **kwargs).to(env.device).eval()
        self.force_aligner_processor = AutoProcessor(self.force_align_model_id_or_path, **kwargs)
        merged_kwargs, kwargs = merge_kwargs(QwenASRKwargs, kwargs)
        self.config = QwenASRKwargs(merged_kwargs)
        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    @torch.inference_mode()
    def _transcribe(
        self,
        audios: list[AudioInput],
        languages: list[str | None],
        contexts: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        pass

    @torch.inference_mode()
    def _align(
        self,
        audios: list[AudioInput] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        emissions: torch.Tensor | None = None,
    ) -> list[tuple[int, list]]:
        pass