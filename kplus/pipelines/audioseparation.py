from __future__ import annotations

import glob
import logging
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from kplus.environment import env
from kplus.tools.config import config
from kplus.tools.progress import MainProgress

if TYPE_CHECKING:
    import torch  # type: ignore
    from demucs.apply import BagOfModels, TensorChunk  # type: ignore
    from demucs.demucs import Demucs  # type: ignore
    from demucs.hdemucs import HDemucs  # type: ignore
    from demucs.htdemucs import HTDemucs  # type: ignore
    Model = Union[Demucs, HDemucs, HTDemucs]

    from .utils import AudioType


logger = logging.getLogger(__name__)

        

class SeparatorMixin:
    def __init__(self, options):
        env.rich  # noqa: B018
        from rich.console import Console # type: ignore  # noqa: I001
        self.console = Console()
        self.model_dir = Path(tempfile.gettempdir()) / "karaoke+models"
        self.repo_url = "https://github.com/Yancovenant/KaraokePlus"
        self.output_dir = "separations_dir"
        self._bootstrapt()

    def _bootstrapt(self):
        env.torch; import torch
        if torch.cuda.is_available():
            env.onnxruntime_gpu
        else:
            env.onnxruntime
        import onnxruntime
        
    @classmethod
    def get_model(cls, options):
        model_map: dict = {"demucs": "mdx_extra_q",
                           "kara": "UVR_MDXNET_KARA_2.onnx",
                           "8kfft": "MDX23C-8KFFT-InstVoc_HQ.ckpt",}
        if options.modelname == "demucs":
            options.modelname = model_map.get(options.modelname)
            return DemucsSeparator(options)
        else:
            options.modelname = model_map.get(options.modelname)
            return MDXSeparator(options)

    def ensureFile(self, target_path: Path):
        """ Ensure path given is a file if not download 
        """
        localpath = self.model_dir / Path(str(target_path)).name
        if localpath.is_file(): return
        env.requests; import requests  # type: ignore  # noqa: B018, I001
        url = f"{self.repo_url.rstrip('/')}/{target_path.lstrip('/')}"
        self.console.print(f"Downloading file from {url}...")
        response = requests.get(url, stream=True, timeout=300)
        if response.status_code == 200:
            size_bytes = int(response.headers.get("content-length", 0))
            pbar = MainProgress(total=size_bytes, unit="B", unit_scale=True, unit_divisor=1024)
            with open(localpath, "wb") as f:
                 for chunk in response.iter_content(chunk_size=8192):
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.pbar.close()
        else:
            raise RuntimeError(f"Failed to download file from {url}, response code: {response.status_code}")
    
    def separate(self, audio: AudioType):
        raise NotImplementedError()

class DemucsSeparator(SeparatorMixin):
    def __init__(self, options):
        super().__init__(options)
        env.diffq, env.demucs  # noqa: B018
        from demucs.pretrained import get_model  # type: ignore
        self.model = get_model(options.modelname).to(env.device).eval()
        self.sr = self.model.samplerate
        self.ac = self.model.audio_channels
        self._populate_model_data(options)

    def _populate_model_data(self, options):
        self.overlap = options.overlap
        self.segment = options.segment
        self.shifts = options.shift

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

    def separate(self, audio: AudioType, external_id: int | None = None):
        import torch  # type: ignore  # noqa: I001
        from .utils import _process_audio
        from demucs.audio import save_audio # type: ignore
        filename = str(Path(audio).stem)
        audio_np = _process_audio(audio, self.sr, self.ac)
        wav = torch.from_numpy(audio_np)
        original_mix: torch.Tensor = wav.clone()
        ref: torch.Tensor = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        segment_length = int(self.sr * self.segment)
        stride = int((1 - self.overlap) * segment_length)
        num_chunks = len(range(0, wav.shape[-1], stride))
        num_models = len(self.model.models) if hasattr(self.model, 'models') else 1
        with MainProgress(total=(num_chunks * num_models) * max(1, self.shifts), desc="Separating...", unit="chunk") as main_bar:
            sources = self._apply_model(self.model, wav[None],
                                        progress=True, shifts=self.shifts,
                                        overlap=self.overlap, segment=self.segment,
                                        pbar=main_bar)[0]
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
        safe_title = "".join([c for c in audio if c.isalpha() or c.isdigit() or c in ' _-']).strip()
        search_pattern = str(Path(config["data_dir"]) / "*" / safe_title)
        matching_dirs = glob.glob(search_pattern)
        if matching_dirs:
            dir_path = Path(matching_dirs[0]).parent
        else:
            filepath = f"{external_id:04d}_{safe_title}_separation" if external_id else f"{safe_title}_separation"
            dir_path = Path(config["data_dir"]) / filepath
            dir_path.mkdir(parents=True, exist_ok=True)
        inst_path = dir_path / f"{self.preset}_{filename}_instrumental.wav"
        save_audio(instruments, str(inst_path), **kwargs)
        vocs_path = dir_path / f"{self.preset}_{filename}_vocs.wav"
        save_audio(vocals, str(vocs_path), **kwargs)
        return SimpleNamespace(sr=self.sr, inst_path=inst_path, vocs_path=vocs_path)


class MDXSeparator(SeparatorMixin):
    def __init__(self, options):
        super().__init__(options)
        target_path = f"/releases/download/v1.0-models/{options.modelname}"
        env.audio_separator; from audio_separator.separator import Separator  # type: ignore  # noqa: B018, I001
        self.separator = Separator(output_dir=self.output_dir)
        self.separator.load_model(model_filename=options.modelname)

    def separate(self, audio):
        if isinstance(audio, Path): audio = str(audio)
        primary_stem, secondary_stem = self.separator.separate(audio)
        vocal_path = os.path.join(self.output_dir, primary_stem if "Vocals" in primary_stem else secondary_stem)
        inst_path = os.path.join(self.output_dir, primary_stem if "Instrumental" in primary_stem else secondary_stem)
        return SimpleNamespace(sr=self.separator.sample_rate, inst_path=inst_path, vocs_path=vocal_path)
    