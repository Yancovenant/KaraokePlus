import logging
from dataclasses import dataclass
from pathlib import Path

from kplus.tools import config, search_for_path

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SeparationResult:
    sr: int
    inst_path: Path
    vocs_path: Path


class SeparatorMixin:
    def __init__(self, modelname: str, **options):
        self.bootstrapt(modelname, **options)

    def bootstrapt(self, modelname: str, **options) -> None:
        raise NotImplementedError()

    def separate(self) -> SeparationResult:
        raise NotImplementedError()

    def make_outdir(self, inputpath: str, external_id: int | None = None) -> None:
        outdir = search_for_path(Path(str(inputpath)).stem)
        if not outdir:
            outdir = (
                f"{external_id:04d}_{inputpath}_separate_data"
                if external_id
                else f"{inputpath}_separate_data"
            )
            outdir = Path(config["data_dir"]).expanduser().resolve() / outdir
            outdir.mkdir(parents=True, exist_ok=True)
        return str(outdir)