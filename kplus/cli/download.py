import sys
import logging

from pathlib import Path

from .command import Command
from kplus.tools import config, RichArgumentParser, safepath
#from kplus.pipelines.songdownloader import SongDownloader
from kplus.pipelines.download import download_song
from .parser_utils import DownloadOptions

logger = logging.getLogger(__name__)

class Download(Command):
    """ Download a YouTube video, extract metadata, and sync lyrics.
    """
    def _parse_config(self, args):
        DownloadOptions.add_options(self.parser, url_only=True)
        self.parser.add_argument("--no-lyrics", action="store_true",
                                  help="Bypass the LRCLIB lookup and only download the media file. Useful for faster execution when metadata is not required.")
        self.parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Custom output directory or exact filename (defaults to temp dir).")
        opt = self.parser.parse_args(args)
        if not opt.url: self.parser.print_help(); sys.exit()
        config.parse_config(opt, setup_logging=True)
        return opt
        
    def run(self, args):
        opt = self._parse_config(args)
        result = download_song(**vars(opt))
        if not opt.no_lyrics and result.lyrics:
            filename = Path(result.filepath).stem
            dirpath = Path(config["data_dir"]) / result.artist / safepath(filename)
            dirpath.mkdir(parents=True, exist_ok=True)
            lyricpath = dirpath / (safepath(filename) + ".txt")
            with open(lyricpath, "w", encoding="utf-8") as f:
                f.write(result.lyrics)
            logger.info("Lyric successfully written to %r", str(lyricpath))