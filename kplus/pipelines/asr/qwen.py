from __future__ import annotations

import logging
import typing as t
from dataclasses import asdict, dataclass

from kplus import env
from kplus.pipelines.utils import ASRResult, TextTiming, WordTiming
from kplus.tools import filter_known_kwargs
from kplus.tools.audio import Audio

from .base import ASRConfig, ASRMixin

if t.TYPE_CHECKING:
    from kplus.pipelines.utils import AudioSegment
    from kplus.tools.audio import AudioNumpy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QwenConfig(ASRConfig):
    max_inference_batch_size: int = -1 # -1 for infinte
    max_new_tokens: int = 8192 #
    num_beams: int = 10

class QwenASR(ASRMixin):
    """ Qwen class ASR """
    _name = "Qwen"
    
    def __init__(self, modelname: str, **options):
        super().__init__(self, **options)
        env.torchvision, env.qwen_asr  # noqa: B018
        from qwen_asr import Qwen3ASRModel  # type: ignore
        self.config = QwenConfig(**options)
        self.model = Qwen3ASRModel.from_pretrained(
            #"Qwen/Qwen3-ASR-1.7B",
            modelname,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
            attn_implementation="sdpa",
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            **options,
            **asdict(self.config),
            forced_aligner_kwargs={
                "dtype": self.config.dtype,
                "device_map": self.config.device_map,
                "attn_implementation": "sdpa",
                **options,
                **asdict(self.config),
            }
        )
        
    def _transcribe(self, audionp: AudioNumpy, audiosegments: list[AudioSegment], reference: str, prg=None, **kwargs) -> list[TextTiming]:
        logger.debug(f"Running Qwen ASR model with {len(audiosegments)} segments and config: {self.sr}")
        audio_chunk_list, results = [], []
        for seg in audiosegments:
            start, end = int(seg.start * self.sr), int(seg.end * self.sr)
            audio_chunk_list.append((audionp[start:end], self.sr))
        logger.debug(f"Prepared {len(audio_chunk_list)} audio chunks for Qwen ASR model")
        batch_result = self.model.transcribe(audio=audio_chunk_list, context=None, return_time_stamps=True, **kwargs)
        logger.debug(f"Qwen ASR model returned {len(batch_result)} segments")
        for seg, aseg in zip(batch_result, audiosegments):
            #prg.update(1)
            if seg.time_stamps is not None:
                words = []
                for word in seg.time_stamps:
                    words.append(
                        WordTiming(
                            start=float(word.start_time + aseg.start),
                            end=float(word.end_time + aseg.start),
                            score=1.0,
                            word=str(word.text)
                        )
                    )
                results.append(
                    TextTiming(
                        words=words,
                        language=seg.language
                    )
                )
        return results

    def _align(self, audionp: AudioNumpy, transcriptions: ASRResult, reference: str, audiosegments: list[AudioSegment], prg=None, **kwargs) -> list[TextTiming]:
        results = []
        task = prg.add_task(description="Aligning...", total=None)
        def progress_callback(seek: float, total: float):
            prg.update(task, completed=seek, total=total)
        audio_chunk_list, text_chunk_list, lang_chunk_list, saved_safe_start = [], [], [], []
        for hyp, seg in zip(transcriptions.texts, audiosegments):
            if not (seg.end < hyp.start or seg.start > hyp.end):
                safe_start = max(0, max(min(hyp.start, seg.start), hyp.start - 1.0) - 0.5)
                safe_end = min(len(audionp), min(max(hyp.end, seg.end), hyp.end + 1.0) + 0.5)
                audio_chunk = Audio.slicenp(audionp, safe_start, safe_end, self.sr)
                assert len(audio_chunk) > 0
                saved_safe_start.append(safe_start)
                audio_chunk_list.append((audio_chunk, self.sr))
                text_chunk_list.append(hyp.latin)
                lang_chunk_list.append(hyp.language)
        align_result = self.model.forced_aligner.align(audio_chunk_list, text_chunk_list, lang_chunk_list)
        assert len(align_result) == len(saved_safe_start)
        for res, safe_start, lang in zip(align_result, saved_safe_start, lang_chunk_list):
            words = []
            for w in res:
                words.append(WordTiming(
                    start=float(round(safe_start + w.start_time, 2)), end=float(round(safe_start + w.end_time,3)),
                    score=2.0, word=str(w.text),
                ))
            results.append(TextTiming(words=words, language=lang))
        return results