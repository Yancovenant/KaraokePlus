from __future__ import annotations

import logging
import os
from pathlib import Path

from kplus import env
from kplus.tools import is_file

from .utils import DownloadError, ErrorType

logger = logging.getLogger(__name__)

class Ytdlp:
    """ Wrapper for yt-dlp module """

    retry_sleep: float = 0.2
    
    FORMAT = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    # Keep this configurable because YouTube extractor clients
    # can change over time.
    player_clients: tuple[str, ...] = (
        "android",
        "web",
    )
    DEFAULT_OPTS = {  # noqa: RUF012
        'format': FORMAT,
        'merge_output_format': 'mp4',
        "outtmpl": "%(title)s.%(ext)s",
        "noplaylist": True,

        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": logger,

        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "continuedl": True,

        "remote_components": ["ejs:github"],
        "extractor_args": {
            "youtube": {"player_clients": list(player_clients)}
        },
        # Keep yt-dlp cache enabled.
        "cachedir": str(Path.home() / ".cache" / "yt-dlp"),
    }

    unrecoverable_error = [  # noqa: RUF012
        ErrorType.UNKNOWN,
        ErrorType.GEO_BLOCKED,
        ErrorType.UNAVAILABLE,
    ]
    
    def __init__(self, cookiefile: str | None, **kwargs) -> None:
        env.yt_dlp  # noqa: B018
        if os.name != "nt": env.deno  # noqa: B018
        self.cookiefile = is_file(cookiefile)
        if self.cookiefile:
            logger.info("Using configured cookie file: %s", self.cookiefile)
        else:
            logger.warning("Cookie file does not exist or not configured: %s", cookiefile,)

    def build_opts(self, outtmpl: str = "%(title)s.%(ext)s", progress_hook = None, *, simulate: bool = False, **kwargs) -> dict:
        opts = self.DEFAULT_OPTS.copy()
        opts.update({
            "outtmpl": outtmpl,
            "simulate": simulate,
            **kwargs,
        })
        if progress_hook:
            opts["progress_hook"] = [progress_hook]
        if self.cookiefile:
            opts["cookiefile"] = str(self.cookiefile)
        return opts

    @property
    def YoutubeDL(self):
        return env.yt_dlp.YoutubeDL

    def extract_info(self, url: str) -> dict:
        with self.YoutubeDL(self.build_opts(simulate=True)) as ydl:
            return ydl.extract_info(url, download=False)

    def download(self, url: str, outtmpl: str, progress_hook) -> Path:
        with self.YoutubeDL(self.build_opts(outtmpl, progress_hook)) as ydl:
            info = ydl.extract_info(url, download=True)
            # yt-dlp's requested format may have a different
            # intermediate extension before merging.
            requested = Path(ydl.prepare_filename(info))
            # Prefer the actual final filename if it exists.
            candidates = [requested, requested.with_suffix(".mp4"),]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            # Last-resort scan.
            parent = requested.parent
            matches = list(parent.glob(f"{requested.stem}.*"))
            for match in matches:
                if match.is_file(): return match
        raise FileNotFoundError("YT-DLP Download complete without producing a single output file: %s", requested)

    def classify_error(self, e: Exception) -> ErrorType:
        failure_type = ErrorType(e)
        if failure_type == ErrorType.AUTH and not self.cookiefile:
            raise DownloadError("Not able to download video youtube as its auth/session error with no `cookiefile`")
        return failure_type