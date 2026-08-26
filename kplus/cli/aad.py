import logging
import sys

from kplus.pipelines import detect_audio_activity, ensure_file, separate_song
from kplus.tools import config
from kplus.tools.audio import Audio

from .command import Command
from .parser_utils import AudioDetectionOptions

logger = logging.getLogger(__name__)

class AAD(Command):
    """ Audio Activity Detection using RMS, Pitch, Harmonics Accoustic, Log Mel
    """
    def _parse_config(self, args):
        AudioDetectionOptions.add_options(self.parser)
        opt = self.parser.parse_args(args)
        if not opt.filepath: self.parser.print_help(); sys.exit()
        config.parse_config(opt, setup_logging=True)
        return opt
        
    def run(self, args):
        opt = self._parse_config(args)
        if opt.separate:
            info = ensure_file(opt.filepath, no_lyrics=True)
            separation_result = separate_song(info, **vars(opt))
            audiopath = separation_result.vocs_path
            sr = separation_result.sr
        else:
            audiopath = opt.filepath
            sr = Audio(str(opt.filepath)).samplerate()
        result = detect_audio_activity(audio=audiopath, sr=sr, **vars(opt))
        