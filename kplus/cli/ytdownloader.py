import sys
import logging

from pathlib import Path

from .command import Command
from kplus.tools.config import config
from kplus.pipelines.songdownloader import SongDownloader

logger = logging.getLogger(__name__)

class YTDownloader(Command):
    """ Download given youtube url """
    def run(self, args):
        self.parser.add_argument("--url", dest="url", help="Youtube Watch/Share URL")
        self.parser.add_argument("--without_lyrics", dest="without_lyrics", action="store_true",
                                 help="Wheter to download youtube video and also include to get the lyrics for it or not (Default: False)")
        self.parser.add_argument("--output", dest="output", help="Output path that the downloaded file would be saved at, (Default: Temporary)")
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.url:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)
        info = SongDownloader(opt.without_lyrics).download(opt.url, opt.output)
        safe_title = "".join([c for c in info.title if c.isalpha() or c.isdigit() or c in ' _-']).strip()
        dir_path = Path(config["data_dir"]) / f"{info.artist}"
        dir_path.mkdir(exist_ok=True, parents=True)
        file_path = dir_path / f"{safe_title}.md"
        markdown_content = (
            f"## Metadata\n"
            f"* **Title:** {info.title}\n"
            f"* **Artist:** {info.artist}\n"
            f"* **Duration:** {info.duration} seconds\n"
            f"\n"
            f"## Local File\n"
            f"* **Filename:** [{info.filename}](./{info.filename})\n"
            f"\n"
            f"## Lyrics\n"
            f"{info.lyrics if info.lyrics else "No lyrics available for this track."}\n"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Metadata file successfully written to: {file_path}")
