import sys
import logging

from pathlib import Path

from .command import Command
from kplus.tools.config import config
from kplus.pipelines import get_track_file, Transcriber

logger = logging.getLogger(__name__)

class Transcribe(Command):
    """ Whisper Transcribe given audio """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path or URL that needs to be make karaoke of, (mp4)")
        self.parser.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="Initial Prompt for whisper")
        group = self.parser.add_argument_group("Advanced Options")
        group.add_argument("--modelname", dest="modelname", default="large-v3", help="Which whisper model used to transcribe")
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)
        info = get_track_file(opt.filepath, opt.lyricsfile is not None)
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = f.readlines()
        filepath = Path(info.filename)
        Transcriber(model_name=opt.modelname).transcribe(filepath, None, info.lyrics)