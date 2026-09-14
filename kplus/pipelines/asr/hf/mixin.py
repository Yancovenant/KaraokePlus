
def ensure_list(data):
    if not isinstance(data, list):
        return [data]
    return data

# ReferenceType: t.TypeAlias = str | list[str] | TextTiming | list[TextTiming]


# class QwenASR(ASRMixin):

#     def get_asr_inputs(
#         self,
#         audios: AudioInput | list[AudioInput],
#         langs: str | list[str],
#         prompts: str | list[str] | None = None,
#         **kwargs
#     ):
#         assert len(langs) == len(audios), f"Different length of audio input: {len(audios)} and langs: {len(langs)}"
#         if prompts is None:
#             prompts = [None] * len(langs)
#         ProcessorKwargs = {
#             "padding": True,
#             "padding_side": "left",
#             "sampling_rate": 16000,
#             "truncation": False,
#             "return_attention_mask": True,
#             "n_window": 50,  # should match config.n_window
#             "return_tensors": "pt"
#         }
#         ProcessorKwargs.update(**kwargs)
#         kwargs = {k: v for k, v in kwargs if k not in ProcessorKwargs}
#         inputs = self.processor.apply_transcription_request(
#             audio=audios,                               # : AudioInput | list[AudioInput],
#             language=langs,                              # : str | list[str] | None = None,
#             prompt=prompts,                                # : str | list[str] | None = None,
#             processor_kwargs=ProcessorKwargs,
#             **kwargs,
#         )
#         inputs.to(self.model.device).to(self.model.dtype)
#         return inputs

#     @torch.no_grad()
#     def _infer(self, inputs, **kwargs):
#         """ Language Model Qwen """
#         return self.model.generate(
#             **inputs,
#             max_new_tokens=self.max_new_tokens,
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
#         )

#     @torch.inference_mode()
#     def _transcribe(
#         self,
#         audios: AudioInput | list[AudioInput],
#         langs: str | list[str],
#     ) -> str:
#         inputs = self.get_asr_inputs(audios, langs, prompts=None)
#         predictions = self._infer(inputs, **kwargs)
#         generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
#         decoded = self.processor.decode(
#             output_ids[:, inputs["input_ids"].shape[1]:],
#             return_format="parsed", # ["raw", "parsed", "transcription_only"]
#             skip_special_tokens=None, # True if `return_format` != `"raw"`
#         )
#         transcriptions = decoded["transcription"]
#         language = decoded["language"]

import logging
from collections import defaultdict

import torch

from kplus.pipelines import detect_language
from kplus.pipelines.utils import ASRResult, AudioSegment, TextTiming
from kplus.tools.audio import Audio, AudioInput, AudioNumpy, AudioType, IndexAudioInput

logger = logging.getLogger(__name__)

class ASRMixin:
    """ Base HF ASR Model Class. """
    model = None
    processor = None
    
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.sr = 16000
        self._load_model(model_name_or_path, **kwargs)

    def _load_model(self, model_name_or_path: str, **kwargs) -> None:
        raise NotImplementedError()

    @torch.inference_mode()
    def _transcribe(self,
        audios: IndexAudioInput,
        languages: list[str | None],
        contexts: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        raise NotImplementedError()

    @torch.inference_mode()
    def _align(self,
        audios: list[AudioInput] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        emissions: torch.Tensor | None = None,
    ) -> list[tuple[int, list]]:
        raise NotImplementedError()

    def inputs(self, audios: list[AudioInput]):
        raise NotImplementedError()

    def detect_language(self, audionp: AudioNumpy, *, seek: float) -> str:
        return detect_language(audionp, seek=seek)

    @torch.no_grad()
    def _infer(self, inputs):
        raise NotImplementedError()

    ## Helper
    def group_by_language(self, audios: list[AudioInput], languages: list[str | None]) -> dict[str | None, IndexAudioInput]:
        groups = defaultdict(list)
        assert len(audios) == len(languages), f"Audios and languages list length missmatch {len(audios)} == {len(languages)}"
        for i, (audio, lang) in enumerate(zip(audios, languages)):
            groups[lang].append((i, audio))
        return groups

    def prepare_data(
        self, audionp: AudioNumpy,
        audiosegments: AudioSegment | list[AudioSegment] | None = None,
        languages: str | list[str] | None = None,
        references: str | list[str] | None = None,
    ) -> tuple[list[AudioInput], list, list, list]:
        """ Return type:: """
        if not audiosegments:
            duration = len(audionp) / self.sr
            audiosegments = [AudioSegment(start=0.0, end=duration)]
        audiosegments = ensure_list(audiosegments)
        languages = ensure_list(languages)
        references = ensure_list(references)

        offsets, langs, audios = [], [], []
        
        languages = languages + [None] * (len(audiosegments) - len(languages))
        assert len(languages) == len(audiosegments), f"Given languages length missmatch {len(languages)} == {len(audiosegments)}"
        references = references + [None] * (len(audiosegments) - len(references))
        assert len(references) == len(audiosegments), f"Given references length missmatch {len(references)} == {len(audiosegments)}"

        for aseg, lang in zip(audiosegments, languages):
            audio_chunk = Audio.slicenp(audionp, aseg.start, aseg.end, self.sr)
            audios.append(audio_chunk)
            offsets.append(aseg.start)
            langs.append(
                lang if lang is not None
                else self.detect_language(audio_chunk, seek=aseg.start)
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
        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"
        for asr_text, offset in zip(results, offsets):
            for word in asr_text.words:
                word.start = word.start + offset
                word.end = word.end + offset
        return ASRResult(texts=results)

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
        assert len(results) == len(offsets), f"produced asr result length missmatch, {len(results)} == len{offsets}"
        for i, ((num_frames, word_spans), offset, lang) in enumerate(zip(results, offsets, langs)):
            audio = audios[i]
            ratio = audio.size(1) / num_frames / self.sr
            parse_timestamp = lambda t, r=ratio: r * t
            assert len(word_spans) == len(hypothesis[i].words)
            for spans, word in zip(word_spans, hypothesis[i].words):
                word.start=parse_timestamp(spans[0].start) + offset
                word.end=parse_timestamp(spans[-1].end) + offset
                word.score=self.make_score(spans)
            if (ori_lang:=hypothesis[i].language) != lang:
                logger.warning(f"Alignment result Language is different: ori {ori_lang} != {lang}")

                
        return ASRResult(texts=hypothesis)


