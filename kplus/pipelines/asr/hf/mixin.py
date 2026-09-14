
import torch
import typing as t
import torchaudio.functional as F

def ensure_list(data):
    if not isinstance(data, list):
        return [data]
    return data

ReferenceType: t.TypeAlias = str | list[str] | TextTiming | list[TextTiming]


class QwenASR(ASRMixin):

    def get_asr_inputs(
        self,
        audios: AudioInput | list[AudioInput],
        langs: str | list[str],
        prompts: str | list[str] | None = None,
        **kwargs
    ):
        assert len(langs) == len(audios), f"Different length of audio input: {len(audios)} and langs: {len(langs)}"
        if prompts is None:
            prompts = [None] * len(langs)
        ProcessorKwargs = {
            "padding": True,
            "padding_side": "left",
            "sampling_rate": 16000,
            "truncation": False,
            "return_attention_mask": True,
            "n_window": 50,  # should match config.n_window
            "return_tensors": "pt"
        }
        ProcessorKwargs.update(**kwargs)
        kwargs = {k: v for k, v in kwargs if k not in ProcessorKwargs}
        inputs = self.processor.apply_transcription_request(
            audio=audios,                               # : AudioInput | list[AudioInput],
            language=langs,                              # : str | list[str] | None = None,
            prompt=prompts,                                # : str | list[str] | None = None,
            processor_kwargs=ProcessorKwargs,
            **kwargs,
        )
        inputs.to(self.model.device).to(self.model.dtype)
        return inputs

    @torch.no_grad()
    def _infer(self, inputs, **kwargs):
        """ Language Model Qwen """
        return self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=None,                     # : GenerationConfig | None = None,
            logits_processor=None,                      # : LogitsProcessorList | None = None,
            stopping_criteria=None,                     # : StoppingCriteriaList | None = None,
            prefix_allowed_tokens_fn=None,              # : Callable[[int, torch.Tensor], list[int]] | None = None,
            synced_gpus=None,                           # : bool | None = None,
            assistant_model=None,                       # : Optional["PreTrainedModel"] = None,
            streamer=None,                              # : Optional["BaseStreamer"] = None,
            negative_prompt_ids=None,                   # : torch.Tensor | None = None,
            negative_prompt_attention_mask=None,        # : torch.Tensor | None = None,
            custom_generate=None,                       # : str | Callable | None = None,
            **kwargs,

            # conversation: list[dict[str, str]] | list[list[dict[str, str]]],
            # chat_template: str | None = None,
            # tools: list[dict] | None = None,
            # documents: list[dict[str, str]] | None = None,
            # add_generation_prompt: bool = False,
            # continue_final_message: bool | str = False,
            # return_assistant_tokens_mask: bool = False,
            # tokenize: bool = False,
            # return_tensors: str | TensorType | None = None,
            # return_dict: bool = False,
            # load_audio_from_video: bool = False,
            # processor_kwargs: dict | None = None,
            # Other Kwargs:
            #   trust_remote_code: None,
            #   cache_implementation: "paged",
            #   input_ids: if inputs is None,
            #   num_beams: int = 1,
            #   max_length: int,
            #   min_length: int,
        )

    @torch.inference_mode()
    def _transcribe(
        self,
        audios: AudioInput | list[AudioInput],
        langs: str | list[str],
    ) -> str:
        inputs = self.get_asr_inputs(audios, langs, prompts=None)
        predictions = self._infer(inputs, **kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            return_format="parsed", # ["raw", "parsed", "transcription_only"]
            skip_special_tokens=None, # True if `return_format` != `"raw"`
        )
        transcriptions = decoded["transcription"]
        language = decoded["language"]


import torch

from transformers import AutoProcessor, AutoModelForCTC, AutoModelForTokenClassification, AutoModelForMultimodalLM

class ASRMixin:
    """ Base HF ASR Model Class. """
    model = None
    processor = None
    
    def __init__(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> None:
        self.sr = 16000
        self._load_model(model_name_or_path, **kwargs)

    def _load_model(self, model_name_or_path, **kwargs) -> None:
        raise NotImplementedError()

    @torch.inference_mode()
    def _transcribe(self, audios: AudioInput | list[AudioInput], audiosegments: AudioSegment | list[AudioSegment]) -> ...:
        raise NotImplementedError()

    @torch.inference_mode()
    def _align(self, audios: AudioInput | list[AudioInput], inputs: torch.Tensor):
        raise NotImplementedError()

    def inputs(self):
        raise NotImplementedError()

    def detect_language(self):
        raise NotImplementedError()

    @torch.no_grad()
    def _infer(self):
        raise NotImplementedError()

    ## Helper
    def prepare_data(
        self,
        audionp: AudioNumpy,
        audiosegments: AudioSegment | list[AudioSegment] | None = None,
        languages: str | list[str] | None = None,
        references: str | list[str] | None = None,
    ) -> tuple[list, list, list, list]:
        """ Return type:: """
        if not audiosegments:
            duration = len(audionp) / self.sr
            audiosegments = [AudioSegment(start=0.0, end=duration)]
        offsets, langs, audios = [], [], []
        audiosegments = ensure_list(audiosegments)
        languages = ensure_list(languages)
        languages = languages + [None] * (len(audiosegments) - len(languages))
        assert len(languages) == len(audiosegments), f"Given languages length missmatch {len(languages)} == {len(audiosegments)}"
        references = ensure_list(references)
        references = references + [None] * (len(audiosegments) - len(references))
        assert len(references) == len(audiosegments), f"Given references length missmatch {len(references)} == {len(audiosegments)}"
        for aseg, lang in zip(audiosegments, languages):
            audio_chunk = Audio.slicenp(audionp, aseg.start, aseg.end, self.sr)
            audios.append(audio_chunk)
            offsets.append(aseg.start)
            langs.append(
                lang if lang is not None
                else self.detect_language(audio_chunk)
            )
        return audios, offsets, langs, references

    @torch.inference_mode()
    def transcribe(
        self,
        audio: AudioType,
        audiosegments: AudioSegment | list[AudioSegment] | None = None,
        contexts: str | list[str] | None = None,
        *,
        return_timestamps: bool = True,
    ) -> ASRResult:
        # In essence transcribe require:
        # - audio
        # - optional: language?, context?
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        audios, offsets, langs, texts = self.prepare_data(
            audionp,
            audiosegments,
            references=contexts,
        )
        results = self._transcribe(
            audios,
            langs,
            texts,
            return_timestamps=return_timestamps,
        )

    @torch.inference_mode()
    def align(
        self,
        audio: AudioType,
        audiosegments: AudioSegment | list[AudioSegment],
        hypothesis: list[TextTiming],
        languages: str | list[str] | None = None,
    ) -> ASRResult:
        # In essence alignment require:
        # - audio, transcript
        # - optional: languages?
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        audios, offsets, langs, texts = self.prepare_data(
            audionp,
            audiosegments,
            languages,
            references=[text.latin for text in hypothesis]
        )
        results = self._align(
            audios=audios,
            transcripts=texts,
            languages=langs,
        )

class ASRConfig:
    _defaults = {
        "processor_kwargs": {
            "return_tensors": "pt",
            "sampling_rate": 16000
        }
    }


class QwenASR(ASRMixin):
    """ Alibaba Qwen ASR Model Class. """
    force_align_model_id_or_path: str = ""
    
    def _load_model(self, model_name_or_path, **kwargs) -> None:
        self.model = AutoModelForMultimodalLM(model_name_or_path, **kwargs)
        self.processor = AutoProcessor(model_name_or_path, **kwargs)
        self.force_aligner = AutoModelForTokenClassification(self.force_align_model_id_or_path, **kwargs)
        self.force_aligner_processor = AutoProcessor(self.force_align_model_id_or_path, **kwargs)
        