import sys
import logging
import os

from pathlib import Path
from urllib.parse import urlparse

from .command import Command
from kplus.tools.config import config
from kplus.pipelines import SongDownloader, SeparationDemucs, VisualizeWaveform, get_track_file
from kplus.environment import env

logger = logging.getLogger(__name__)

class Separate(Command):
    """ Separate either input URL or file path into 2 different stems (vocals, instrumentals) """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath", help="The input file path or URL that needs to be separated")
        stem_choices = ['inst', 'vocs', 'all']
        self.parser.add_argument("-n", '--stems', dest="stems", choices=stem_choices, default="all", help="Specify the stems output. Accepted values: %s." % (stem_choices,))
        group = self.parser.add_argument_group("Advanced Options")
        preset_choices = ["turbo", "fast", "standard", "high", "studio"]
        group.add_argument("-p", "--preset", dest="preset", choices=preset_choices, default="high", help="Specify the presets choice, Accepted values: %s." % (preset_choices,))
        group.add_argument("--segment", dest="segment", type=int,
                           help="Length (in seconds) of audio should be processed at once, !!! This can help save memory of graphic card")
        group.add_argument("--overlap", dest="overlap", type=int,
                           help="The overlap between the splits")
        group.add_argument("--shifts", dest="shifts", type=int,
                           help="""if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
                                and apply the oppositve shift to the output. This is repeated `shifts` time and
                                all predictions are averaged. This effectively makes the model time equivariant
                                and improves SDR by up to 0.2 points.""")
        group.add_argument("--visualize", action="store_true", dest="visualize", help="Visualize the separated stems into 5 different graphic, (Waveform, Mel Spectogram, Harmonic vs Percussive, Pitch Tracking (F0), Chromagram)")
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)
        info = get_track_file(Path(opt.filepath), True)
        filepath = info.filename
        separation_model = SeparationDemucs(preset=opt.preset, overlap_ratio=opt.overlap,
                                            segment_size=opt.segment, shifts=opt.shifts)
        separation_info = separation_model.separate(filepath, opt.stems)
        logger.info(f"Finished separating {filepath}")
        del separation_model.model, separation_model
        env.clean()
        if opt.visualize:
            for path in [separation_info.inst_path, separation_info.vocs_path]:
                if os.path.exists(path):
                    VisualizeWaveform().visualize(path)