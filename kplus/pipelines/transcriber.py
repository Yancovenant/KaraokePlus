from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.tools.progress import MainProgress

from .aad import AudioSegment
from .utils import AudioLoader, Result, Segment, WordTiming

if TYPE_CHECKING:
    from .utils import AudioType


logger = logging.getLogger(__name__)

QWEN_CONTEXT_PROMPT = context = """
This is a short clip of a highly dynamic song.
1. Your job is to Transcribe EXACTLY what is sung in this specific audio clip. Do not hallucinate.
2. The clip may contains extremly repetitive vocal chants. Do not summarize or skip repetitions. If a word is sung 15 times, you must output it exactly 15 times.
3. The clip may features non-standard vocalizations. Specifically, transcribe the chant as 'oheh'.
4. The lyrics may rapidly code-switch between English, Japanese (e.g., '行こう'), Spanish (e.g., 'dale'), and French (e.g., 'allez').
5. The clip may contain ad-libs, and dropped sung lyrics, you must include it exactly as heard.

This is the full lyrics transcription:
"""

class Transcriber:
    def __init__(self, verbose: bool = False):
        self.sr = 16000 # Whisper and qwen both used 16K sample rate
        self.verbose = verbose

    def load_model(self, modelname, **kwargs):
        if modelname == "qwen":
            env.torchvision, env.qwen_asr  # noqa: B018
            from qwen_asr import Qwen3ASRModel  # type: ignore # noqa: I001
            import torch # type: ignore
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                and torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
            device_map = f"cuda:{'1' if torch.cuda.device_count() > 1 else '0'}" if env.device.type == "cuda" else env.device.type
            self.model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B",
                dtype=dtype,
                device_map=device_map,
                attn_implementation="sdpa",
                max_inference_batch_size=-1, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
                max_new_tokens=8192, # Maximum number of tokens to generate. Set a larger value for long audio input.
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs={"dtype": dtype,
                                        "device_map": device_map,
                                        "attn_implementation": "sdpa",},)
        else:
            env.stable_ts, env.faster_whisper  # noqa: B018
            import stable_whisper  # type: ignore
            self.beamsize = kwargs.pop("beamsize")
            self.max_threads = max(1, kwargs.pop("max_threads"))
            compute_type = "float16" if env.device.type == "cuda" else "float32"
            self.model = stable_whisper.load_faster_whisper(
                modelname, device=env.device.type,
                compute_type=compute_type, num_workers=self.max_threads,
                **kwargs)
        return self.model

    def transcribe(self,
                   audio: AudioType,
                   audio_segments: list[AudioSegment] | None = None,
                   reference: str | None = None,
                   sr: int | None = None) -> Result:
        audio_loader = AudioLoader(audio, samplerate=self.sr, channels=1)
        audio = audio_loader.audio_np
        if not audio_segments:
            duration = len(audio) / self.sr
            audio_segments = [AudioSegment(start=0.0, end=duration)]
        try:
            results = []
            with MainProgress(total = len(audio_segments), desc="Starting Transcriptions...", unit="chunk") as main_bar:
                if self.model.__module__.startswith('faster_whisper.'):
                    # normal transcribe expect str, float() timestamp
                    time_batches = ",".join(f"{seg.start},{seg.end}" for seg in audio_segments)
                    # batch transcribe expect a dict?
                    # time_batches = [{"start": seg.start, "end": seg.end} for seg in audio_segments]
                    batch_result = self.model.transcribe(audio, language=None, clip_timestamps=time_batches,
                                                        initial_prompt=reference, beam_size=self.beamsize, #batch_size=1, # need to be None
                                                        repetition_penalty=1.2, condition_on_previous_text=False)
                    for res in batch_result:
                        main_bar.update(1)
                        seg_words = []
                        for w in res.words:
                            seg_words.append(WordTiming(
                                start=float(w.start), end=float(w.end),
                                score=float(w.probability), word=str(w.word)))
                        results.append(Segment(words=seg_words, language=batch_result.language))
                elif self.model.__module__.startswith('qwen_asr.'):
                    logger.debug(f"Running Qwen ASR model with {len(audio_segments)} segments and config: {self.sr}")
                    audio_chunk_list = []
                    for seg in audio_segments:
                        start, end = int(seg.start * self.sr), int(seg.end * self.sr)
                        audio_chunk_list.append((audio[start:end], self.sr))
                    logger.debug(f"Prepared {len(audio_chunk_list)} audio chunks for Qwen ASR model")
                    
                    batch_result =  self.model.transcribe(audio=audio_chunk_list, context=None, return_time_stamps=True,)
                    logger.debug(f"Qwen ASR model returned {len(batch_result)} segments")
                    for seg, aseg in zip(batch_result, audio_segments):
                        main_bar.update(1)
                        if seg.time_stamps is not None:
                            seg_words = []
                            for word in seg.time_stamps:
                                seg_words.append(WordTiming(
                                    start=float(word.start_time + aseg.start), end=float(word.end_time + aseg.start),
                                    score=1.0, word=str(word.text)))
                            results.append(Segment(words=seg_words, language=seg.language))
                else:
                    raise RuntimeError(f"model not supported: {self.model.__module__}")
        except Exception as err:
            logger.exception(f"Error while doing transcriptions, error: {err}")
            raise
        results.sort(key=lambda x: x.words[0].start if x.words else 0.0)
        return Result(segments=results)
