import logging
import typing as t

import torch
import torchaudio.functional as F
from transformers import AutoModelForCTC, AutoProcessor

from kplus import env
from kplus.pipelines.utils import TextTiming, WordTiming
from kplus.tools.audio import AudioInput, IndexAudioInput

from ..kwargs_utils import ASRKwargs, merge_kwargs
from .mixin import ASRMixin

logger = logging.getLogger(__name__)


class Wav2Vec2Kwargs(ASRKwargs):
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

class Wav2Vec2(ASRMixin):
    """ Meta Facebook ASR Model Class. """
    def _load_model(self, model_name_or_path: str, **kwargs) -> None:
        self.model = AutoModelForCTC.from_pretrained(model_name_or_path, **kwargs).to(env.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
        merged_kwargs, kwargs = merge_kwargs(Wav2Vec2Kwargs, kwargs)
        self.config = Wav2Vec2Kwargs(merged_kwargs)
        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    def inputs(self, audios: list[AudioInput]):
        return self.processor(audio=audios, sampling_rate=self.sr, **self.config["processor_kwargs"]).to(device=self.model.device)

    @torch.no_grad()
    def _infer(self, inputs) -> t.Any:
        return self.model(**inputs)

    @torch.inference_mode()
    def _transcribe(
        self,
        audios: list[AudioInput],
        languages: list[str | None],
        contexts: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        """ Model Transcribe Inference """
        # contexts?
        audio_groups: dict[str | None, IndexAudioInput] = self.group_by_language(audios, languages)
        asr_results = [None] * sum(len(items) for items in audio_groups.values())
        for lang, items in audio_groups.items():
            indices = [i for i,_ in items]
            audio_group = [audio for _,audio in items]
            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)
            inputs = self.inputs(audio_group)
            outputs = self._infer(inputs)
            logits = outputs.logits
            generated_token_ids = logits.argmax(dim=-1)
            transcriptions: list[str] = self.processor.batch_decode(generated_token_ids)
            result = [
                TextTiming(
                    words=[
                        WordTiming(start=None, end=None, score=None, word=word)
                        for word in transcript.split()
                    ],
                    language=lang
                )
                for transcript in transcriptions
            ]
            if return_timestamps:
                emissions = logits.log_softmax(dim=-1)
                align_results = self._align(emissions=emissions, transcripts=transcriptions)
                for i, (num_frames, word_spans) in enumerate(align_results):
                    ratio = audio_group[i].size(1) / num_frames / self.sr
                    parse_timestamp = lambda t, r=ratio: r * t
                    assert len(word_spans) == len(result[i].words)
                    for spans, word in zip(word_spans, result[i].words):
                        word.start=parse_timestamp(spans[0].start)
                        word.end=parse_timestamp(spans[-1].end)
                        word.score=self.make_score(spans)
            assert len(result) == len(indices)
            for i, res in zip(indices, result):
                asr_results[i] = res
        logger.debug(asr_results)
        assert len(asr_results) == len(languages), f"transcript result length missmatch {len(asr_results)} == {len(languages)}"
        return asr_results

    def make_score(self, spans) -> float:
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)
        
    @torch.inference_mode()
    def _align(
        self,
        audios: list[AudioInput] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        emissions: torch.Tensor | None = None,
    ) -> list[tuple[int, list]]:
        def _compute_alignment(emissions_list: torch.Tensor, refs: list[str]) -> list:
            assert len(emissions_list) == len(refs), f"Emission and ref length missmatch, {len(emissions_list)} == {len(refs)}"
            norm_func = str.lower if "mms" in self.model.name_or_path else str.upper
            norm_refs = [norm_func(ref) for ref in refs]
            # `torchaudio.functional` alignment support batch == 1 only.
            assert len(norm_refs) == emissions_list.size(0) # Size 0 is [batch]
            results = []
            for i, norm_ref in enumerate(norm_refs):
                token_ids = self.processor.tokenizer(norm_ref, **self.config["tokenizer_kwargs"])["input_ids"].to(dtype=torch.long, device=self.model.device)
                # Remove spaces
                token_ids = [t for t in token_ids if t not in (self.processor.tokenizer.word_delimiter_token_id,)]
                emissions = emissions_list[i].unsqueeze(0)
                logger.debug(f"[{i}] Aligning Emissions {emissions.shape}\nTarget: {self.processor.batch_decode(token_ids)}")
                alignments, scores = F.forced_align(emissions, token_ids, blank=self.processor.tokenizer.pad_token_id)
                scores = scores.exp()
                token_spans = F.merge_tokens(alignments[0], scores[0], blank=self.processor.tokenizer.pad_token_id)
                # Remove spaces
                token_spans = [s for s in token_spans if s.token not in (self.processor.tokenizer.word_delimiter_token_id,)]

                # Flatten
                words = norm_ref[i].split()
                n = [len(w) for w in words]
                assert sum(n) == len(token_spans), f"Length Missmatch: Token {len(token_spans)} | Words {sum(n)}"
                j, word_spans = 0, []
                for l in n:
                    word_spans.append(token_spans[j:j+l])
                    j+=l
                num_frames = emissions.size(1)
                results.append((num_frames, word_spans)) # word_spans == list[TokenSpan(start, end, token, score)]
            # Returning list[list[tuple[int, TokenSpan]]]
            # since we can't calculate its timestamp without enough data
            return results

        # If `audios` is pass group by lang
        # Else adapter already loaded
        if audios is not None:
            audio_groups: dict[str | None, IndexAudioInput] = self.group_by_language(audios, languages)
            results = [None] * sum(len(items) for items in audio_groups.values())
            for lang, items in audio_groups.items():
                indices = [i for i,_ in items]
                audio_group = [audio for _,audio in items]
                transcript_group = [transcripts[i] for i in indices]
                self.processor.tokenizer.set_target_lang(lang)
                self.model.load_adapter(lang)
                inputs = self.inputs(audio_group)
                outputs = self._infer(inputs)
                logits = outputs.logits
                emissions = logits.log_softmax(dim=-1)
                group_results = _compute_alignment(emissions, transcript_group)
                for i, global_i in enumerate(indices):
                    results[global_i] = group_results[i]
        elif emissions is not None:
            return _compute_alignment(emissions, transcripts)
        raise ValueError("Either `audios` or `emissions` must be provided")
