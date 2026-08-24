import logging
import os
import sys
from pathlib import Path

from kplus.environment import env
from kplus.pipelines import SeparatorMixin, VisualizeWaveform, get_track_file
from kplus.tools.config import config
from kplus.tools.rich import RichArgumentParser, RichHelpFormatter

from .command import Command
from .download import Download

logger = logging.getLogger(__name__)

class Separate(Command):
    """ Separate either input URL or file path into 2 different stems (vocals, instrumentals) """

    @staticmethod
    def internal_separate_options(parser: RichArgumentParser):
        group = parser.add_argument_group("Separate Options")
        #stem_choices = ['inst', 'vocs', 'all']
        #group.add_argument("-n", '--stems', dest="stems", choices=stem_choices, default="all",
        #                   help="Specify which stems to output: 'inst' (Instrumental), 'vocs' (Vocals), or 'all' (Both).")
        #group.add_argument("modelname", type=str, metavar="MODEL", help="Override the default separation model (e.g., demucs).")
        group.add_argument("--visualize", action="store_true", dest="visualize",
                           help="Generate 5 visual analysis graphs for the stems (Waveform, Mel Spectrogram, Harmonic/Percussive, Pitch Tracking, Chromagram).")
        
    @staticmethod
    def shared_separate_options(parser: RichArgumentParser):
        subparsers = parser.add_subparsers(
            dest="separation_modeltype", prog=parser.prog, # need to be added to not break RichHelpFormatter
            title="Separation Engines",
            description="Select the model backend to perform separation. Run 'separate <engine> --help' for engine-specific flags.",
            help="Model choice",
        )
        # Demucs
        demucs_parser = subparsers.add_parser("demucs", help="Demucs architecture engine.")
        demucs_parser.add_argument("separation_modelname", nargs="?", default="mdx_extra_q", metavar="MODEL",
                           help="Demucs model checkpoint to load (default: mdx_extra_q).")
        demucs_parser.add_argument("--segment", dest="segment", type=int, metavar="SECONDS",
                           help="Process audio in chunks of this length. Crucial for saving GPU VRAM on lower-end hardware.")
        demucs_parser.add_argument("--overlap", dest="overlap", type=float, metavar="RATIO",
                           help="The overlap percentage between audio splits (e.g., 0.25). Higher overlap reduces boundary artifacts but increases processing time.")
        demucs_parser.add_argument("--shifts", dest="shifts", type=int, metavar="N",
                           help="Number of random time-shifts. Improves SDR (Signal-to-Distortion) by up to 0.2 points, but multiplies processing time by N.")    
        
        uvr_parser = subparsers.add_parser("uvr", help="Ultimate Vocal Remover / MDX architecture engine.")
        uvr_parser.add_argument(
            "separation_modelname", default="UVR_MDXNET_KARA_2.onnx", metavar="MODEL",
            help="UVR model file to load (default: UVR_MDXNET_KARA_2.onnx)."
        )
        #group = parser.add_argument_group("Shared Separate Options", "inherited by any command that triggers a separation")
        #group = group.add_argument_group("Advanced Options")
        #preset_choices = ["turbo", "fast", "standard", "high", "studio"]
        #group.add_argument("-p", "--preset", dest="preset", choices=preset_choices, default="high",
        #                   help="Select a quality preset. Higher quality takes longer to process, Accepted values: %s." % (preset_choices,))
        
    def _parse_config(self, args):
        self.internal_separate_options(self.parser)
        self.shared_separate_options(self.parser)
        Download.target_options(self.parser)
        opt = self.parser.parse_args(args)
        if not opt.filepath: self.parser.print_help(); sys.exit()
        config.parse_config(opt, setup_logging=True)
        return opt
        
    def run(self, args):
        opt = self._parse_config(args)
        info = get_track_file(opt.filepath, True)
        filepath = Path(info.filename).expanduser()
        separation_model = SeparatorMixin.from_model(**vars(opt))
        separation_info = separation_model.separate(filepath)
        logger.info(f"Finished separating {filepath}")
        logger.info(f">> Inst Path: {separation_info.inst_path}")
        logger.info(f">> Vocs Path: {separation_info.vocs_path}")
        logger.info(f">> SR: {separation_info.sr}")
        del separation_model.model, separation_model
        env.clean()
        if opt.visualize:
            for path in [separation_info.inst_path, separation_info.vocs_path]:
                if os.path.exists(path):
                    VisualizeWaveform().visualize(path)