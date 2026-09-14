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
            "padding": True,
            "padding_side": "left",
            "sampling_rate": 16000,
            "truncation": False,
            "return_attention_mask": True,
            "n_window": 50,  # should match config.n_window
            "return_tensors": "pt"
        },
        "generation_kwargs": {
            "max_new_tokens": 8192,
            "num_beams": 10,
            
            # max_new_tokens=self.max_new_tokens,
#             generation_config=None,                     # : GenerationConfig | None = None,
#             logits_processor=None,                      # : LogitsProcessorList | None = None,
#             stopping_criteria=None,                     # : StoppingCriteriaList | None = None,
#             prefix_allowed_tokens_fn=None,              # : Callable[[int, torch.Tensor], list[int]] | None = None,
#             synced_gpus=None,                           # : bool | None = None,
#             assistant_model=None,                       # : Optional["PreTrainedModel"] = None,
#             streamer=None,                              # : Optional["BaseStreamer"] = None,
#             negative_prompt_ids=None,                   # : torch.Tensor | None = None,
#             negative_prompt_attention_mask=None,        # : torch.Tensor | None = None,
#             custom_generate=None,                       # : str | Callable | None = None,
#             **kwargs,

#             # conversation: list[dict[str, str]] | list[list[dict[str, str]]],
#             # chat_template: str | None = None,
#             # tools: list[dict] | None = None,
#             # documents: list[dict[str, str]] | None = None,
#             # add_generation_prompt: bool = False,
#             # continue_final_message: bool | str = False,
#             # return_assistant_tokens_mask: bool = False,
#             # tokenize: bool = False,
#             # return_tensors: str | TensorType | None = None,
#             # return_dict: bool = False,
#             # load_audio_from_video: bool = False,
#             # processor_kwargs: dict | None = None,
#             # Other Kwargs:
#             #   trust_remote_code: None,
#             #   cache_implementation: "paged",
#             #   input_ids: if inputs is None,
#             #   num_beams: int = 1,
#             #   max_length: int,
#             #   min_length: int,
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
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name_or_path, **kwargs).to(env.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
        self.force_aligner = AutoModelForTokenClassification.from_pretrained(self.force_align_model_id_or_path, **kwargs).to(env.device).eval()
        self.force_aligner_processor = AutoProcessor.from_pretrained(self.force_align_model_id_or_path, **kwargs)
        merged_kwargs, kwargs = merge_kwargs(QwenASRKwargs, kwargs)
        self.config = QwenASRKwargs(merged_kwargs)
        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    def inputs(
        self,
        audios: list[AudioInput],
        langs: list[str | None],
        prompts: list[str | None],
    ):
        return self.processor.apply_transcription_request(
            audio=audios,
            language=langs,
            prompt=prompts,
            processor_kwargs=self.config["processor_kwargs"]
        ).to(device=self.model.device)

    @torch.no_grad()
    def _infer(self, inputs):
        return self.model.generate(
            **inputs,
            **self.config["generation_kwargs"]
        )

    @torch.inference_mode()
    def _transcribe(
        self,
        audios: list[AudioInput],
        languages: list[str | None],
        contexts: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        inputs = self.inputs(audios, languages, prompts=contexts)
        outputs = self._infer(inputs)
        generated_token_ids = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            return_format="parsed",                     # ["raw", "parsed", "transcription_only"]
            skip_special_tokens=None,                   # True if `return_format` != `"raw"`
        )
        logger.debug("Decoded", decoded)
        transcriptions = decoded["transcription"]
        languages = decoded["language"]

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