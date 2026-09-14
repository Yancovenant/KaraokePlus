import torch, copy

from .mixin import ASRMixin
from ..kwargs_utils import ASRKwargs, merge_kwargs

from kplus import env
from kplus.tools import rich


class Wav2Vec2Kwargs(ASRKwargs):
    _defaults = {
        "processor_kwargs": {
            "return_tensors": "pt",
            "padding": True,
        }
    }

class Wav2Vec2(ASRMixin):
    """ Meta Facebook ASR Model Class. """
    def _load_model(self, model_name_or_path, **kwargs) -> None:
        self.model = AutoModelForCTC.from_pretrained(model_name_or_path, **kwargs).to(env.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
        merged_kwargs, kwargs = merge_kwargs(Wav2Vec2Kwargs, kwargs)
        self.config = Wav2Vec2Kwargs(merged_kwargs)
        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    def inputs(self, audios: AudioInput) -> torch.Tensor:
        inputs = self.processor(audio=audios, sampling_rate=self.sr, **self.config["processor_kwargs"])
        inputs.to(device=self.model.device)
        return inputs

    @torch.no_grad()
    def _infer(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(**inputs)

    @torch.inference_mode()
    def _transcribe(
        self,
        audios: AudioInput,
        languages: list[str | None],
        contexts: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        """ Model Transcribe Inference """
        # languages? contexts?
        inputs = self.processor(audio=audios, **self.config.processor_kwargs)
        inputs.to(device=self.model.device, dtype=self.model.dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        generated_token_ids = logits.argmax(dim=-1)
        transcriptions = self.processor.batch_decode(generated_token_ids)
        logger.debug(rich.Panel(transcriptions))

        if not return_timestamps:
            return [
                TextTiming(words=[
                    WordTiming(word=word)
                ], language=lang)
                for transcript, lang in zip(transcriptions, languages)
                for word in transcript.split()
            ]

        # Force Alignment
        emissions = logits.log_softmax(dim=-1)
        # plot_emission
        align_results = self._align(transcriptions, languages, inputs=inputs, emissions=emissions)
        results = []
        for res in align_results:
            words = []
            for words_res in res:
                words.append(WordTiming(
                    start=words_res["start"],
                    end=words_res["end"],
                    score=words_res["score"],
                    word=words_res["word"],
                ))
            results.append(TextTiming(words=words, language=""))
        
    @torch.inference_mode()
    def _align(
        self,
        audios: AudioInput | None = None,
        inputs: torch.Tensor | None = None,
        transcripts: list[str] | ... = ...,
        languages: list[str] | ... = ...,
        *,
        emissions: torch.Tensor | None = None,
    ) -> list[list[dict]]:
        if audios is None and inputs is None:
            raise ValueError("Either `audios` or `inputs` param must be passed")
        if inputs is None:
            inputs = self.get_inputs(audios)
        if emissions is None:
            logits = self._infer(inputs).logits
            emissions = logits.log_softmax(dim=-1)

        ref_norm = str.lower if "mms" in self.model.name_or_path else str.upper
        ref_texts = [ref_norm(text) for text in transcripts]
        token_ids: list = self.processor.tokenizer(ref_texts)["input_ids"]
        targets = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        
        alignments, scores = F.forced_align(emissions, targets, blank=self.processor.tokenizer.pad_token_id)
        scores = scores.exp()
        token_span_lists = F.merge_tokens(alignments[0],scores[0], blank=self.processor.tokenizer.pad_token_id)
        #token_spans = [s for s in token_spans if s.token not in (processor.tokenizer.word_delimiter_token_id,)]
        word_lists = [[text.split()] for text in texts]
        n_word_lists = [[len(w)] for text in word_lists for w in text]
        assert all(sum(n_words) == len(token_spans) for n_words, token_spans in zip(n_word_lists, token_span_lists)), f"{len(token_spans)} real spans vs {sum(n_words)} expected chars -- don't trust output"
        word_spans_list = []
        for token_spans, n_words in zip(token_span_lists, n_word_lists):
            assert len(token_spans) == sum(n_words)
            i, clusters = 0, []
            for l in n_words:
                clusters.append(token_spans[i: i + l])
                i += l
            word_spans_list.append(clusters)
        results = []
        for word_spans, words in zip(word_spans_list, word_lists):
            words_res = []
            for spans, w in zip(word_spans, words):
                # spans is list of words??
                words_res.append({
                    "start": spans[0].start,
                    "end": spans[-1].end,
                    "score": self._score(spans),
                    "word": w,
                })
                # plot_audio
            results.append(words_res)
        # plot score
        # plot alignment
        return results