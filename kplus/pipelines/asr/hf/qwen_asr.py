import logging

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
)

from kplus import env
from kplus.pipelines.utils import ASRResult, TextTiming, WordTiming
from kplus.tools.audio import AudioInput, AudioNumpy

from ..utils import QWEN_LANGUAGES, get_default_dtype
from .kwargs_utils import ASRKwargs, merge_kwargs
from .mixin import ASRMixin
from .wav2vec2 import Wav2Vec2

logger = logging.getLogger(__name__)


class QwenASRKwargs(ASRKwargs):
    _defaults = {  # noqa: RUF012
        "processor_kwargs": {
            "padding": True,
            "sampling_rate": 16000,
            "return_tensors": "pt"
        },
        "generation_kwargs": {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
        },
        "common_kwargs": {
            "max_inference_batch_size": 20,
            "dtype": get_default_dtype(),
            "device_map": (
                "cuda:" + 
                ("1" if torch.cuda.device_count() > 1 else "0")
            ) if env.device.type == "cuda" else env.device.type
        }
    }

class QwenASR(ASRMixin):
    """ Alibaba Qwen ASR Model Class. """
    force_align_model_id_or_path: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    
    def _load_model(self, model_name_or_path, **kwargs) -> None:
        merged_kwargs, kwargs = merge_kwargs(QwenASRKwargs, kwargs)
        self.config = QwenASRKwargs(merged_kwargs)
        self.max_batch_size = self.config["common_kwargs"].pop("max_inference_batch_size", 32)
        
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name_or_path, **self.config["common_kwargs"]).to(env.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **self.config["common_kwargs"])

        self.force_aligner = Wav2Vec2("facebook/mms-1b-all", **self.config["common_kwargs"])

        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    def detect_language(self, audionp: AudioNumpy, *, seek: float) -> str:
        lang = super().detect_language(audionp, seek=seek)
        return QWEN_LANGUAGES.get(lang, "en")
    
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
        ).to(device=self.model.device, dtype=self.model.dtype)

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
        batch_size = self.max_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(audios)
        outs: list[str] = []
        for i in range(0, len(audios), batch_size):
            sub_audios = audios[i: i + batch_size]
            sub_languages = languages[i: i + batch_size]
            sub_contexts = contexts[i: i + batch_size]
            inputs = self.inputs(sub_audios, sub_languages, prompts=sub_contexts)
            outputs = self._infer(inputs)
            decoded = self.processor.decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                return_format="transcription_only",         # ["raw", "parsed", "transcription_only"]
                skip_special_tokens=None,                   # True if `return_format` != `"raw"`
            )
            outs.extend(list(decoded))
            del inputs, outputs, decoded
        assert len(outs) == len(languages)
        results: list[TextTiming] = []
        for text, lang in zip(outs, languages):
            words = []
            for word in text.split():
                words.append(WordTiming(start=None, end=None, score=None, word=word))
            results.append(TextTiming(words=words, language=lang))
        if return_timestamps:
            align_results = self._align(
                audios=audios,
                transcripts=[res.text for res in results],
                languages=languages
            )
            # Needs to return list[TextTiming]
            return [res.text for res in align_results.texts]
        return results

    @torch.inference_mode()
    def _align(
        self,
        audios: list[AudioInput] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        emissions: torch.Tensor | None = None,
    ) -> list[tuple[int, list]]:
        return self.force_aligner.align(audios, transcripts, languages)