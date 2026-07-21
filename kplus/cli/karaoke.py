import sys
import logging
import os

from pathlib import Path
from urllib.parse import urlparse

from .command import Command
from kplus.tools.config import config
from kplus.pipelines import SongDownloader, SeparationDemucs, VisualizeWaveform, get_track_file, \
    Aligner, AAD
from kplus.environment import env

logger = logging.getLogger(__name__)

class Karaoke(Command):
    """ Separate either input URL or file path into 2 different stems (vocals, instrumentals) """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path or URL that needs to be make karaoke of, (mp4)")
        self.parser.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="If input is not URL, and no lyrics path were given, default to multiplex only")
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)
        info = get_track_file(opt.filepath, opt.lyrics is not None)
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = f.readlines()
        filepath = Path(info.filename)
        separation_model = SeparationDemucs(overlap_ratio=0.75,
                                            segment_size=200, shifts=1)
        separation_info = separation_model.separate(filepath, "all")
        logger.info(f"Finished separating {filepath}")
        del separation_model.model, separation_model
        env.clean()
        audio_segments = AAD(False).get_audio_segments(separation_info.vocal_tensor,
                sr=separation_info.sr)
        aligner_model = Aligner()
        aligner_info = aligner_model.main(separation_info.vocal_tensor,
                separation_info.sr, info.lyrics, audio_segments)