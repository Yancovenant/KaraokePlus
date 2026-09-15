import logging
from collections import defaultdict

import torch

from kplus.pipelines.utils import ASRResult, AudioSegment, TextTiming
from kplus.tools.audio import Audio, AudioInput, AudioNumpy, AudioType, IndexAudioInput

logger = logging.getLogger(__name__)

def ensure_list(data):
    if not isinstance(data, list):
        return [data]
    return data

class ASRMixin:
    """ Base HF ASR Model Class. """
    model = None
    processor = None
    
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.sr = 16000
        self.lid_model = None
        self._load_model(model_name_or_path, **kwargs)

    def _load_model(self, model_name_or_path: str, **kwargs) -> None:
        raise NotImplementedError()

    @torch.inference_mode()
    def _transcribe(self,
        audios: list[AudioInput],
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
    ) -> list[tuple[int, list]]:
        raise NotImplementedError()

    def inputs(self, audios: list[AudioInput]):
        raise NotImplementedError()

    def detect_language(self, audionp: AudioNumpy, *, seek: float) -> str:
        if not self.lid_model:
            from kplus.pipelines.asr import BaseASR
            self.lid_model = BaseASR.from_model(whisper="large-v3", extra_models=[])
        return self.lid_model.detect_language(audionp, seek=seek)

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
        self.lid_model = None
        return audios, offsets, langs, references

    @torch.inference_mode()
    def transcribe(
        self,
        audio: AudioType,
        audiosegments: AudioSegment | list[AudioSegment] | None = None,
        contexts: str | list[str] | None = None,
        *,
        languages: str | list[str] | None = None,
        return_timestamps: bool = True,
    ) -> ASRResult:
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        audios, offsets, langs, texts = self.prepare_data(
            audionp,
            audiosegments,
            languages=languages,
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
                word.start = word.start + offset if word.start is not None else offset
                word.end = word.end + offset if word.end is not None else offset
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


