from __future__ import annotations  # noqa: I001

from kplus.tools.audio import Audio, AudioType, AudioNumpy
# Need to be below
import torch
import typing as t
import logging
import difflib
import copy

from dataclasses import field, dataclass

from kplus import env
from kplus.tools import rich, get_phonetic
from kplus.pipelines.utils import TextTiming, ASRResult

from .utils import get_default_dtype

if t.TYPE_CHECKING:
    from kplus.pipelines.utils import AudioSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ASRConfig:
    dtype: torch.dtype = field(default_factory=get_default_dtype)
    device_map: str = (
        "cuda:" + 
        ("1" if torch.cuda.device_count() > 1 else "0")
    ) if env.device.type == "cuda" else env.device.type
    

class ASRMixin:
    """ ASR Mixin """
    def __init__(self, **options):
        if "dtype" in options:
            dtype = options.pop("dtype").lower()
            if dtype != "auto":
                try:
                    options["dtype"] = getattr(torch, dtype)
                except AttributeError:
                    raise ValueError("dtype not recognize %s", dtype)
        self.sr = 16000 # Used for all model

    def _transcribe(self, audio: AudioNumpy, audiosegments: list[AudioSegment], reference: str, prg=None, **kwargs) -> list[TextTiming]:
        raise NotImplementedError()

    def transcribe(self, audio: AudioType, audiosegments: list[AudioSegment], reference: str, **kwargs) -> ASRResult:
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        if not audiosegments:
            duration = len(audionp) / self.sr
            audiosegments = [AudioSegment(start=0.0, end=duration)]
        try:
            with rich.make_progress(is_download=False) as prg:
                prg.add_task(description=f"{self._name} Starting ASR...", total=None)
                results = self._transcribe(audionp, audiosegments, reference, prg, **kwargs)
        except Exception as err:
            logger.exception(f"Error while doing transcriptions, error: {err}")
            raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        return ASRResult(texts=results)

    def _fix_duplicate(self, new_res, ori: TextTiming):
        new_words = [get_phonetic(w.word.strip()).latin for w in new_res.words]
        ori_words = [get_phonetic(w.word.strip()).latin for w in ori.words]
        patched = []
        matcher = difflib.SequenceMatcher(None, ori_words, new_words) # should this converted to a number for faster performance? like jiwer does it
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' or tag == 'replace': patched.extend(new_res.words[j1:j2])
            elif tag == 'delete':
                logger.warning(f"  -> Whisper deleting words {i1} - {i2}")
                for missing_idx in range(i1, i2):
                    patched.append(ori.words[missing_idx])
            elif tag == 'insert':
                logger.warning(f"  -> Dropping Whisper hallucination: {[w.word for w in new_res.words[j1:j2]]}")
        new_res.words = patched
        return new_res

    def _align(self, audionp: AudioNumpy, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], prg=None, **kwargs) -> list[TextTiming]:
            raise NotImplementedError()
    
    def align(self, audio: AudioType, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], **kwargs) -> list[TextTiming]:
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        assert len(audiosegments) > 0
        try:
            with rich.make_progress(is_download=False) as prg:
                prg.add_task(description=f"{self._name} Starting Alignment...", total=None)
                results = self._align(audionp, transcriptions, reference, audiosegments, prg, **kwargs)
        except Exception as err:
            logger.exception(f"Error while doing Alignment, error: {err}")
            raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        return ASRResult(texts=results)


class MMS_FA(ASRMixin):
    """ mms_fa facebook wav2vec2 aligner """
    _name = "MMS FA"
    
    def __init__(self, modelname: str, **options):
        super().__init__(**options)
        env.torchaudio; import torchaudio  # type: ignore # noqa: B018, I001
        bundle = torchaudio.pipelines.MMS_FA
        self.sr = bundle.sample_rate
        self.model = bundle.get_model(with_star=True).to(env.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def _tokenize(self, text: str) -> list:
        tokens = []
        for chunk in text.split():
            if not chunk.strip(): continue
            tokens.append({"original": chunk, "token": get_phonetic(chunk).latin})
        return tokens

    def _align(self, audionp: AudioNumpy, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], prg=None, **kwargs) -> list[TextTiming]:
        results = []
        task = prg.add_task(description="Aligning...", total=None)
        def progress_callback(seek: float, total: float):
            prg.update(task, completed=seek, total=total)
        duration = len(audionp) / self.sr
        for hyp, seg in zip(transcriptions.texts, audiosegments):
            if not (seg.end < hyp.start or seg.start > hyp.end):
                safe_start = max(0, max(min(hyp.start, seg.start), hyp.start - 1.0) - 0.5)
                safe_end = min(duration, min(max(hyp.end, seg.end), hyp.end + 1.0) + 0.5)
                audio_chunk = Audio.slicenp(audionp, safe_start, safe_end, self.sr)
                assert len(audio_chunk) > 0
                audio_chunk = torch.from_numpy(audio_chunk).unsqueeze(0)
                tokens = self._tokenize(hyp.text)
                transcript_tokens = ["*"]
                for tok in tokens:
                    transcript_tokens.extend(list(tok["token"]))
                    transcript_tokens.append("*")
                try:
                    with torch.inference_mode():
                        emission, _ = self.model(audio_chunk.to(env.device))
                        token_spans = self.aligner(emission[0], self.tokenizer(transcript_tokens))
                except Exception as err:
                    logger.error(f"!!! Error while doing ctc align: {err}", exc_info=True)
                    raise
                char_spans = [span for token, span in zip(transcript_tokens, token_spans) if token != "*"]
                ratio = audio_chunk.size(1) / emission.size(1)
                del token_spans, emission, audio_chunk
                char_idx, words = 0, []
                for i, tok in enumerate(tokens):
                    word_len = len(tok["token"])
                    current_char_spans = char_spans[char_idx : char_idx + word_len]
                    char_idx += word_len
                    if not current_char_spans: continue
                    first_char_span = current_char_spans[0]
                    last_char_span = current_char_spans[-1]
                    local_start = int(ratio * first_char_span[0].start) / self.sr
                    local_end = int(ratio * last_char_span[-1].end) / self.sr
                    total_score = sum(c[0].score for c in current_char_spans) / len(current_char_spans)
                    rw = copy.copy(hyp.words[i])
                    rw.start = local_start + safe_start
                    rw.end = local_end + safe_start
                    rw.score = float(total_score)
                    words.append(rw)
                results.append(TextTiming(words=words, language=hyp.language))
                prg.update(task)
        return results