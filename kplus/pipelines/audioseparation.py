from __future__ import annotations

import logging
import random
import glob

from types import SimpleNamespace
from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional, Dict, Any

from kplus.environment import env
from kplus.tools.progress import MainProgress
from kplus.tools.config import config

if TYPE_CHECKING:
    import torch
    
    from demucs.apply import BagOfModels, TensorChunk
    from demucs.htdemucs import HTDemucs
    from demucs.demucs import Demucs
    from demucs.hdemucs import HDemucs
    Model = Union[Demucs, HDemucs, HTDemucs]


logger = logging.getLogger(__name__)


class SeparationDemucs:
    def __init__(self, overlap_ratio: Optional[float] = None,
                 segment_size: Optional[int] = None,
                 shifts: Optional[int] = None,
                 preset: Optional[str] = "high"):
        env.diffq, env.demucs # Diffq first then demucs
        from demucs.pretrained import get_model

        self.device = env.device
        self.model = get_model("mdx_extra_q").to(self.device).eval()
        self.sr = self.model.samplerate
        self.ac = self.model.audio_channels
        self.shifts, self.overlap, self.segment = self._preprocess_preset(preset)
        if any(val is not None for val in [overlap_ratio, segment_size, shifts]):
            if preset:
                logger.warning("Applying custom options while using preset's is not compatible. Overriding the preset template...")
                preset = "custom"
        self.preset = preset
        if overlap_ratio is not None:
            self.overlap = overlap_ratio
        if segment_size is not None:
            self.segment = segment_size
        if shifts is not None:
            self.shifts = shifts
        logger.debug(f"Running separation | Preset: {preset} | shift: {self.shifts} | "
                     f"Overlap: {self.overlap:.2f} | segment: {self.segment}")
    
    def _preprocess_preset(self, preset: Optional[str] = "high"):
        env.torch
        import torch
        preset_map = {
            "turbo": {"overlap": 0.1, "shift": 0,},
            "fast": {"overlap": 0.15, "shift": 1,},
            "standard": {"overlap": 0.3, "shift": 2,},
            "high": {"overlap": 0.45, "shift": 2},
            "studio": {"overlap": 0.75, "shift": 5,}}
        settings = preset_map.get(preset, preset_map["high"]).copy()
        logger.debug(f"Getting max preset for device: {self.device}")
        try:
            if self.device.type == "cuda": # add xpu later?
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                free_gb = free_bytes / (1024**3)
                logger.debug("Running separation on CUDA VRAM: %.2f GB" % free_gb)
                min_seg, max_seg = 15.0, 200.0
                multiplier = 22.0
            else: # cpu
                try:
                    env.psutil
                    import psutil
                    free_gb = psutil.virtual_memory().available / (1024**3)
                    logger.debug("Running separation on CPU VRAM: %.2f GB" % free_gb)
                except Exception as err:
                    logger.warning("!!! Failed to get free vram on cpu device: %s", str(err))
                    free_gb = 6.0 # REVIEW: need to check this
                if self.device.type == "cpu": # add mps later?
                    min_seg, max_seg = 10.0, 45.0
                    multiplier = 8.0
            calculated_size = free_gb * multiplier
            final_capped_seg = float(max(min_seg, min(max_seg, calculated_size)))
        except Exception as err:
            logger.warning("!!! Exception while getting the resource for separation preset: %s", str(err))
            final_capped_seg = 15.0 if self.device.type == "cpu" else 30.0
        return settings["shift"], settings["overlap"], final_capped_seg

    def _apply_model(self,
            model: Union[BagOfModels, Model],
            mix: Union[torch.Tensor, TensorChunk],
            shifts: int = 1, split: bool = True,
            overlap: float = 0.25, transition_power: float = 1.,
            progress: bool = False, segment: Optional[float] = None,
            pbar=None, model_idx: str = "") -> torch.Tensor:
        """ Code purely from demucs file
        """
        env.demucs, env.torch
        from demucs.utils import center_trim, DummyPoolExecutor
        from demucs.apply import BagOfModels, TensorChunk, tensor_chunk
        from demucs.htdemucs import HTDemucs
        import torch
        
        pool = DummyPoolExecutor()
        kwargs: Dict[str, Any] = {'shifts': shifts, 'split': split,
                                  'overlap': overlap, 'transition_power': transition_power,
                                  'progress': progress, 'segment': segment,
                                  'pbar': pbar, 'model_idx': model_idx}
        out: Union[float, torch.Tensor]
        if isinstance(model, BagOfModels):
            # Special treatment for bag of model.
            # We explicitely apply multiple times `apply_model` so that the random shifts
            # are different for each model.
            estimates: Union[float, torch.Tensor] = 0.
            totals = [0.] * len(model.sources)
            total_models = len(model.models)
            for i, (sub_model, model_weights) in enumerate(zip(model.models, model.weights)):
                #original_model_device = next(iter(sub_model.parameters())).device
                sub_model.to(self.device)
                kwargs['model_idx'] = f"{i + 1}/{total_models}"
                out = self._apply_model(sub_model, mix, **kwargs)
                #sub_model.to(original_model_device)
                for k, inst_weight in enumerate(model_weights):
                    out[:, k, :, :] *= inst_weight
                    totals[k] += inst_weight
                estimates += out
                del out

            assert isinstance(estimates, torch.Tensor)
            for k in range(estimates.shape[1]):
                estimates[:, k, :, :] /= totals[k]
            return estimates

        model.to(self.device).eval()
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
                shifted_out = self._apply_model(model, shifted, **kwargs)
                out += shifted_out[..., max_shift - offset:]
            out /= shifts
            assert isinstance(out, torch.Tensor)
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
            weight = torch.cat([torch.arange(1, segment_length // 2 + 1, device=self.device),
                            torch.arange(segment_length - segment_length // 2, 0, -1, device=self.device)])
            assert len(weight) == segment_length
            # If the overlap < 50%, this will translate to linear transition when
            # transition_power is 1.
            weight = (weight / weight.max())**transition_power
            futures = []
            for offset in offsets:
                chunk = TensorChunk(mix, offset, segment_length)
                future = pool.submit(self._apply_model, model, chunk, **kwargs)
                futures.append((future, offset))
                offset += segment_length
            if progress:
                from tqdm import tqdm
                desc_text = f"   ↳ Model {model_idx}" if model_idx else "   ↳ Processing"
                futures = tqdm(futures, unit_scale=scale,
                               ncols=120, unit='seconds',
                               desc=desc_text, dynamic_ncols=True, position=1)
            for future, offset in futures:
                chunk_out = future.result()
                chunk_length = chunk_out.shape[-1]
                out[..., offset:offset + segment_length] += (
                    weight[:chunk_length] * chunk_out).to(mix.device)
                sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
                if pbar:
                    pbar.update(1)
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
            padded_mix = mix.padded(valid_length).to(self.device)
            with torch.no_grad():
                out = model(padded_mix)
            assert isinstance(out, torch.Tensor)
            return center_trim(out, length)
    
    def separate(self, input_path: Path, stems: str = "inst", external_id: Optional[int] = None):
        env.torch
        from demucs.audio import save_audio
        from .utils import load_audio
        filename = str(Path(input_path).stem)
        wav = load_audio(input_path, self.sr, self.ac)
        original_mix: torch.Tensor = wav.clone()
        ref: torch.Tensor = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        segment_length = int(self.sr * self.segment)
        stride = int((1 - self.overlap) * segment_length)
        num_chunks = len(range(0, wav.shape[-1], stride))
        num_models = len(self.model.models) if hasattr(self.model, 'models') else 1
        with MainProgress(total=(num_chunks * num_models) * max(1, self.shifts), desc="Separating %s" % filename, unit="chunk") as main_bar:
            sources = self._apply_model(self.model, wav[None], progress=True, shifts=self.shifts,
                                        overlap=self.overlap, segment=self.segment, pbar=main_bar)[0]
        sources *= ref.std()
        sources += ref.mean()
        sources = list(sources)
        vocals = sources.pop(self.model.sources.index("vocals"))
        instruments = original_mix - vocals
        del sources
        kwargs = {'samplerate': self.sr,
                  'bitrate': 320, 'preset': 2,
                  'clip': "rescale", 'as_float': False,
                  'bits_per_sample': 16,}
        inst_path, vocs_path = None, None
        safe_title = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ' _-']).strip()
        search_pattern = str(Path(config["data_dir"]) / "*" / safe_title)
        matching_dirs = glob.glob(search_pattern)
        if matching_dirs:
            dir_path = Path(matching_dirs[0]).parent
        else:
            filepath = f"{external_id:04d}_{safe_title}_separation" if external_id else f"{safe_title}_separation"
            dir_path = Path(config["data_dir"]) / filepath
            dir_path.mkdir(parents=True, exist_ok=True)
        if stems in ["inst", "all"]:
            inst_path = dir_path / f"{self.preset}_{filename}_instrumental.wav"
            save_audio(instruments, str(inst_path), **kwargs)
        if stems in ["vocs", "all"]:
            vocs_path = dir_path / f"{self.preset}_{filename}_vocs.wav"
            save_audio(vocals, str(vocs_path), **kwargs)
        return SimpleNamespace(vocal_tensor=vocals, sr=self.sr, inst_path=inst_path, vocs_path=vocs_path)


class AudioSeparation:
    DEFAULT_DEMUCS_OPTS = {
        "shifts": None,
        "segment_size": None,
        "overlap_ratio": None,
        "preset": "high"
    }
    def __init__(self, modelname: str):
        pass