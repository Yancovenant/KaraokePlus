"""
Main Entry Point for Song Downloader
"""
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from kplus import env
from kplus.tools import rich

from .lyrics import Lrclib
from .utils import DownloadError, ErrorType, LyricsError
from .ytdlp import Ytdlp

logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class DownloadResult:
    title: str
    artist: str
    duration: float
    lyrics: str | None
    filepath: str

    def print(self) -> None:
        table = rich.Table.grid(expand=True, padding=(0, 1))
        table.add_row("Artist", self.artist)
        table.add_row("Title", self.title)
        table.add_row("Duration", f"{self.duration:.0f}s")
        table.add_row("Lyrics", "Found" if self.lyrics else "Skipped")
        table.add_row("File", str(self.filepath))
        out = rich.Panel(table, padding=1, title="Karaoke+ Download Complete", border_style="color(14)")
        logger.info(out)

class Downloader:
    """ Main class for SongDownloader """
    def __init__(self, cookiefile: str | None = None, **kwargs):
        self.session = env.requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) '
                'AppleWebKit/537.36 '
                '(KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })
        self.max_attempts = kwargs.pop("max_attempts", 4)
        self.downloader = Ytdlp(cookiefile, **kwargs)
        self.lyrics_api = Lrclib(self.session, **kwargs)

    def _extract_info(self, url: str) -> tuple[str, str, str]:
        info = self.downloader.extract_info(url)
        title = info.get("title", "Unknown")
        artist = info.get("artist", info.get("uploader", "Unknown"))
        duration = float(info.get("duration", 0))
        return title, artist, duration

    def _download(self, url: str, outtmpl: str, prg) -> str:
        # with rich.make_progress(is_download=True) as prg:
        task = prg.add_task("Downloading", total=None)
        def hook(data):
            if data.get("status") == "downloading":
                total = data.get("total_bytes", data.get("total_bytes_estimate", 0))
                downloaded = data.get("downloaded_bytes")
                if total:
                    prg.update(task, total=total, completed=downloaded, description="Downloading...")
            elif data.get("status") == "finished":
                total = data.get("total_bytes", data.get("total_bytes_estimate", 0))
                prg.update(task, total=total or 1, completed=total or 1, description="Download Complete.")
        filepath = self.downloader.download(url, outtmpl, progress_hook=hook)
        prg.update(task, advance=1, description="Downloaded.")
        return str(filepath)

    def get_lyrics(self, *info) -> str:
        return self.lyrics_api.get_lyrics(*info)

    def make_output(self, outpath: str | None, eid: int | None = None) -> str:
        default_filename = "%(title)s.%(ext)s"
        filename = (
            f"{eid:04d}_{default_filename}"
            if eid is not None
            else default_filename
        )
        if not outpath:
            return str(Path(tempfile.gettempdir()) / filename)
        target = Path(outpath).expanduser().resolve()
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            return str(target)
        target.mkdir(parents=True, exist_ok=True)
        return str(target / filename)

    def download(
        self,
        url: str,
        outpath: str | None = None,
        external_id: int | None = None,
        *, no_lyrics: bool = False
    ) -> DownloadResult:
        output = self.make_output(outpath, external_id)
        last_error = None
        with rich.make_progress(is_download=True) as prg:
            task = prg.add_task("Preparing...", total=2 if no_lyrics else 3)
            for i in range(1, self.max_attempts + 1):
                try:
                    logger.info(
                        "Download attempt (%d/%d): %s",
                        i, self.max_attempts, url
                    )
                    prg.update(task, description="Extracting Metadata...")
                    title, artist, duration = self._extract_info(url)
                    prg.update(task, advance=1)
                    lyrics = None
                    if not no_lyrics:
                        prg.update(task, description="Searching Lyrics...")
                        lyrics = self.get_lyrics(title, artist, duration)
                        prg.update(task, advance=1)
                        if not lyrics: raise LyricsError("Lyrics not found with sufficient confidence.")
                    prg.update(task, description="Downloading...")
                    filepath = self._download(url, output, prg=prg)
                    prg.update(task, advance=1, description="Completed.")
                    result = DownloadResult(
                        title=title,
                        artist=artist,
                        duration=duration,
                        lyrics=lyrics,
                        filepath=filepath,
                    )
                    if env.verbose:
                        result.print()
                    return result
                except LyricsError:
                    raise
                except Exception as e:
                    last_error = e
                    failure_type, delay = self._handle_error(i, e)
                    if i >= self.max_attempts: break
                    prg.update(task, description=f"{failure_type} - retrying in {delay:.1f}")
                    time.sleep(delay)
        raise DownloadError(f"Download failed after {self.max_attempts:d} attempts : {last_error!s}") from last_error

    def _handle_error(self, attempt: int, e: Exception):
        failure_type = self.downloader.classify_error(e)
        logger.warning("Download failure: type=%s, error=%s", failure_type, e, exc_info=True)  # noqa: LOG014
        if failure_type in self.downloader.unrecoverable_error:
            raise DownloadError(f"{failure_type}: {e!s}") from e
        delay = self.downloader.retry_sleep * (2 ** (attempt - 1))
        if failure_type == ErrorType.RATE_LIMIT:
            return failure_type, min(60, delay)
        elif failure_type == ErrorType.NETWORK:
            return failure_type, min(30, delay)
        return failure_type, self.max_attempts
    