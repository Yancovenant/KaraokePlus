from __future__ import annotations

import logging
import random
import typing as t
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kplus import env
from kplus.tools import filter_known_kwargs, rich, safepath
from kplus.tools.audio import Audio, AudioTensor

from .base import SeparationResult, SeparatorMixin

if t.TYPE_CHECKING:
    from demucs.apply import (
        BagOfModels as DemucsBagOfModels,  # type: ignore
    )
    from demucs.apply import Model as DemucsModel
    from demucs.apply import TensorChunk

logger = logging.getLogger(__name__)

class DemucsSeparator(SeparatorMixin):
    def bootstrapt(self, modelname: str, **options):
        logger.info("Chosen Separator Class: (Demucs)")
        env.torch, env.diffq, env.demucs  # noqa: B018
        from demucs.pretrained import get_model as _get_model_demucs  # type: ignore
        self.overlap = options.pop("overlap", 0.75)
        self.segment = options.pop("segment", 30)
        self.shifts = options.pop("shifts", 1)
        self.num_workers = options.pop("num_workers", 0)
        demucs_params, options = filter_known_kwargs(_get_model_demucs, options)
        with rich.make_progress(is_download=False) as prg:
            prg.add_task("Loading Demucs Model...", total=None)
            self.model = _get_model_demucs(modelname, **demucs_params).to(env.device).eval()
        self.sr = self.model.samplerate
        self.ac = self.model.audio_channels
        table = rich.Table.grid(rich.Column(), rich.Column(ratio=1), expand=True, padding=(0, 0))
        table.add_row("Model", ": " + modelname)
        table.add_row("Segment", ": " + str(self.segment))
        table.add_row("Overlap", ": " + str(self.overlap))
        table.add_row("Shifts", ": " + str(self.shifts))
        table.add_row("SampleRate", ": " + str(self.sr))
        table.add_row("AudioChannels", ": " + str(self.ac))
        table.add_row("NumWorkers", ": " + str(self.num_workers))
        logger.debug(rich.Panel(table, title="Demucs Configuration", padding=1))

    def _preprocess(self, inputpath: str) -> tuple[AudioTensor, AudioTensor, AudioTensor]:
        audio = Audio(inputpath, samplerate=self.sr, channels=self.ac)
        wav = audio.tensor
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8
        return wav, mean, std

    def _post_process(self, inputpath: str, vocals: AudioTensor, instruments: AudioTensor, outdir: str) -> tuple[str, str]:
        from demucs.audio import save_audio  # type: ignore
        save_kwargs = {
            'samplerate': self.sr,
            'bitrate': 320,
            'preset': 2,
            'clip': "rescale",
            'as_float': False,
            'bits_per_sample': 16,
        }
        outfilename = safepath(inputpath)
        outdir = Path(outdir)
        inst_path = str(outdir / f"[S={self.shifts}|O={self.overlap}|SZ={self.segment}]_inst_{outfilename}.wav")
        vocs_path = str(outdir / f"[S={self.shifts}|O={self.overlap}|SZ={self.segment}]_vocs_{outfilename}.wav")
        save_audio(instruments, str(inst_path) **save_kwargs)
        save_audio(vocals, str(vocs_path), **save_kwargs)
        return inst_path, vocs_path


    def _apply_model(self,
        model: DemucsBagOfModels | DemucsModel,
        mix: AudioTensor | TensorChunk,
        shifts: int = 1,
        split: bool = True,
        overlap: float = 0.25,
        transition_power: float = 1.,
        progress: bool = False,
        num_workers: int = 0,
        pool=None,
        segment: float | None = None,
        prg=None,
        model_idx: str = ""
    ) -> AudioTensor:
        """ Demucs ``apply_model`` implementation
        """
        import torch  # type: ignore
        from demucs.apply import BagOfModels as DemucsBagOfModels  # type: ignore
        from demucs.apply import TensorChunk, tensor_chunk
        from demucs.htdemucs import HTDemucs  # type: ignore
        from demucs.utils import (  # type: ignore
            DummyPoolExecutor,
            center_trim,
        )

        if pool is None:
            if num_workers > 0 and env.device.type == 'cpu':
                pool = ThreadPoolExecutor(num_workers)
            else:
                pool = DummyPoolExecutor()
        kwargs: dict[str, t.Any] = {
            'shifts': shifts,
            'split': split,
            'overlap': overlap,
            'transition_power': transition_power,
            'progress': progress,
            'pool': pool,
            'segment': segment,
            "prg": prg,
            'model_idx': model_idx
        }
        out: float | AudioTensor
        res: float | AudioTensor

        if isinstance(model, DemucsBagOfModels):
            # Special treatment for bag of model.
            # We explicitely apply multiple times `apply_model` so that the random shifts
            # are different for each model.
            estimates: float | AudioTensor = 0.
            totals = [0.] * len(model.sources)
            total_models = len(model.models)
            for i, (sub_model, model_weights) in enumerate(zip(model.models, model.weights)):
                sub_model.to(env.device)
                kwargs['model_idx'] = f"{i + 1}/{total_models}"
                res = self._apply_model(sub_model, mix, **kwargs)
                out = res
                for k, inst_weight in enumerate(model_weights):
                    out[:, k, :, :] *= inst_weight
                    totals[k] += inst_weight
                estimates += out
                del out
            assert isinstance(estimates, AudioTensor)
            for k in range(estimates.shape[1]):
                estimates[:, k, :, :] /= totals[k]
            return estimates
        model.to(env.device).eval()
        assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
        batch, channels, length = mix.shape
        if shifts:
            kwargs['shifts'] = 0
            max_shift = int(0.5 * model.samplerate)
            mix = tensor_chunk(mix)
            assert isinstance(mix, TensorChunk)
            padded_mix = mix.padded(length + 2 * max_shift)
            out = 0.
            for _ in range(shifts):
                offset = random.randint(0, max_shift)
                shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
                res = self._apply_model(model, shifted, **kwargs)
                shifted_out = res
                out += shifted_out[..., max_shift - offset:]
            out /= shifts
            assert isinstance(out, AudioTensor)
            return out
        elif split:
            kwargs['split'] = False
            out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
            sum_weight = torch.zeros(length, device=mix.device)
            if segment is None:
                segment = model.segment
            assert segment is not None and segment > 0.
            segment_length: int = int(model.samplerate * segment)
            stride = int((1 - overlap) * segment_length)
            offsets = range(0, length, stride)
            scale = float(format(stride / model.samplerate, ".2f"))
            # We start from a triangle shaped weight, with maximal weight in the middle
            # of the segment. Then we normalize and take to the power `transition_power`.
            # Large values of transition power will lead to sharper transitions.
            weight = torch.cat([torch.arange(1, segment_length // 2 + 1, device=env.device),
                            torch.arange(segment_length - segment_length // 2, 0, -1, device=env.device)])
            assert len(weight) == segment_length
            # If the overlap < 50%, this will translate to linear transition when
            # transition_power is 1.
            weight = (weight / weight.max())**transition_power
            futures = []
            for offset in offsets:
                chunk = TensorChunk(mix, offset, segment_length)
                future = pool.submit(self._apply_model, model, chunk, **kwargs)
                futures.append((future, offset))
            task = None
            if progress and prg:
                total_seconds = len(futures) * scale
                desc_text = f"Model {model_idx}.." if model_idx else 'Separating..'
                task = prg.add_task(description=desc_text, total=total_seconds)
                # futures = rich.track(futures, description=f"Model {model_idx}.." if model_idx else 'Separating..')
                #from tqdm import tqdm
                #desc_text = f"   ↳ Model {model_idx}" if model_idx else "   ↳ Processing"
                #futures = tqdm(futures, unit_scale=scale,
                #                ncols=120, unit='seconds',
                #                desc=desc_text, dynamic_ncols=True, position=0)
            for future, offset in futures:
                try:
                    chunk_out = future.result()
                except BaseException:
                    logger.warning("Keyboard Int, closing all separating process")
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                chunk_length = chunk_out.shape[-1]
                out[..., offset:offset + segment_length] += (
                    weight[:chunk_length] * chunk_out).to(mix.device)
                sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
                if task and prg:
                    prg.update(task, advance=scale)
            assert sum_weight.min() > 0
            out /= sum_weight
            assert isinstance(out, torch.Tensor)
            return out
        else:
            valid_length: int
            if isinstance(model, HTDemucs) and segment is not None:
                valid_length = int(segment * model.samplerate)
            elif hasattr(model, 'valid_length'):
                valid_length = model.valid_length(length)  # type: ignore
            else:
                valid_length = length
            mix = tensor_chunk(mix)
            assert isinstance(mix, TensorChunk)
            padded_mix = mix.padded(valid_length).to(env.device)
            with torch.no_grad():
                out = model(padded_mix)
            assert isinstance(out, AudioTensor)
            return center_trim(out, length)
            

    def _process(self, mix: AudioTensor, mean: AudioTensor, std: AudioTensor) -> tuple[AudioTensor, AudioTensor]:
        with rich.make_progress(is_download=False) as prg:
            prg.add_task("Separating...", total=None)
            out = self._apply_model(
                self.model, ((mix - mean) / std)[None],
                progress=True, shifts=self.shifts,
                overlap=self.overlap, segment=self.segment,
                prg=prg, num_workers=self.num_workers,
            )
        out = out * std + mean
        res = dict(zip(self.model.sources, out[0]))
        vocals = res["vocals"]
        instruments = mix - vocals
        del res
        return vocals, instruments
    
    def separate(self, inputpath: str, external_id: int | None = None) -> SeparationResult:
        outdir = self.make_outdir(inputpath, external_id)
        wav, mean, std = self._preprocess(inputpath)
        vocals, instruments = self._process(wav, mean, std)
        inst_path, vocs_path = self._post_process(inputpath, vocals, instruments, outdir)
        return SeparationResult(sr=self.sr, inst_path=inst_path, vocs_path=vocs_path)
    