import logging
import sys

from kplus.pipelines import VisualizeWaveform, ensure_file, separate_song
from kplus.tools import config, rich

from .command import Command
from .parser_utils import SeparateOptions

logger = logging.getLogger(__name__)

class Separate(Command):
    """ Separate either input URL or file path into 2 different stems (vocals, instrumentals) """

    def _parse_config(self, args):
        SeparateOptions.add_options(self.parser)
        self.parser.add_argument("--visualize", action="store_true", dest="visualize",
                                 help="Generate 5 visual analysis graphs for the stems (Waveform, Mel Spectrogram, Harmonic/Percussive, Pitch Tracking, Chromagram).")
        opt = self.parser.parse_args(args)
        if not opt.filepath: self.parser.print_help(); sys.exit()
        if opt.demucs and opt.uvr:
            raise ValueError("Cannot use Demucs model and UVR model at the same time: %s - %s", opt.demucs, opt.uvr)
        if not opt.demucs and not opt.uvr:
            raise ValueError("Either ``--demucs MODEL`` or ``--uvr MODEL`` need's to be passed at runtime")
        config.parse_config(opt, setup_logging=True)
        return opt
        
    def run(self, args):
        opt = self._parse_config(args)
        info = ensure_file(opt.filepath, no_lyrics=True)
        result = separate_song(info, **vars(opt))

        table = rich.Table.grid(rich.Column(), rich.Column(ratio=1), expand=True, padding=(0,0))
        table.add_row("Inst Path", ": " + str(result.inst_path))
        table.add_row("Vocs Path", ": " + str(result.vocs_path))
        table.add_row("SR", result.sr)
        logger.info(rich.Panel(table, title=f"Finished separating {info.filepath!s}"))
        
        if opt.visualize:
            for path in [result.inst_path, result.vocs_path]:
                if path.is_file():
                    VisualizeWaveform().visualize(path)
