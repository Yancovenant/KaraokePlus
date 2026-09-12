import torch
import torchaudio.functional as F


class CTCMixin:
    def __init__(self, model_id: str, model: AutoModelForCTC, processor: AutoProcessor, **kwargs):
        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.sr = 16000

    def _score(self, spans):
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

    def _unflatten(list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

    @torch.inference_mode()
    def _align(self, audio_chunk: AudioNumpy, hypothesis: TextTiming):
        inputs = self.processor(audio_chunk, return_tensors="pt", sampling_rate=self.sr)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        emissions = logits.log_softmax(dim=-1)
        reference = hypothesis.latin.lower() if "mms" in self.model_id else hypothesis.latin.upper()
        input_ids = self.processor.tokenizer(reference)["input_ids"]
        targets = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
        alignments, scores = F.forced_align(emissions, targets, blank=self.processor.tokenizer.pad_token_id)
        scores = scores.exp()
        token_spans = F.merge_tokens(alignments[0], scores[0], blank=self.processor.tokenizer.pad_token_id)
        token_spans = [s for s in token_spans if s.token not in (self.processor.tokenizer.word_delimiter_token_id,)]

        words = hypothesis.latin.split()
        n = [len(w) for w in words]
        assert sum(n) == len(token_spans), "word_lens out of sync with flat token ids -- check tokenizer behavior before trusting output"

        word_spans = self._unflatten(token_spans, n)
        del inputs, logits, emissions, reference, input_ids, targets, alignments, token_spans, words, scores
        env.clean()
        return word_spans

    def align(self, audio: AudioType, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], **options):
        audionp = Audio()
        pass


