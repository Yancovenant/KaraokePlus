from __future__ import annotations  # noqa: I001

from kplus.tools.audio import Audio, AudioType, AudioNumpy
# Need to be below
import torch
import typing as t
import logging
import difflib

from dataclasses import field, dataclass, asdict

from kplus import env
from kplus.tools import rich, get_phonetic
from kplus.pipelines.utils import TextTiming, WordTiming, ASRResult

from .utils import get_default_dtype

if t.TYPE_CHECKING:
    from kplus.pipelines.utils import AudioSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ASRConfig:
    dtype: torch.dtype = field(default_factory=get_default_dtype)
    device_map: str = (
        "cuda:"
        "1" if torch.cuda.device_count() > 1 else "0"
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
    
    def align(self, audio: AudioType, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], prg=None, **kwargs) -> list[TextTiming]:
        audionp = Audio(audio, samplerate=self.sr, channels=1).numpy
        if not audiosegments:
            duration = len(audionp) / self.sr
            audiosegments = [AudioSegment(start=0.0, end=duration)]
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