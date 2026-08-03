import logging
import sys

from kplus.environment import env
from kplus.pipelines import AAD as _AAD
from kplus.tools.config import config

from .command import Command

logger = logging.getLogger(__name__)

class AAD(Command):
    """ Audio Activity Detection using RMS, Pitch, Harmonics Accoustic
    """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path (.wav)")
        self.parser.add_argument("--visualize", dest="verbose", action="store_true")
        self.parser.add_argument("--precision_ms", default=0.5, type=float)
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)

        env.demucs; from demucs.audio import AudioFile # type: ignore  # noqa: B018, I001
        sr = AudioFile(str(opt.filepath)).samplerate()
        logger.debug(f"Successfully get sr: {sr}")
        opt.sr = sr
        filtered_audio_np, audio_segments = _AAD(opt).get_audio_segments(opt.filepath)
