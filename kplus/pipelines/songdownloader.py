from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from kplus.environment import env
from kplus.tools.progress import MainProgress, SubProgress
from kplus.tools.rich import rich

logger = logging.getLogger(__name__)

class LyricsError(Exception):
    """Raised when lyrics cannot be resolved with sufficient confidence."""


class YTDLPError(Exception):
    """Raised when yt-dlp cannot complete the requested operation."""
    

@dataclass
class YTDLPEnvironment:
    cookie_file: Path | None = None
    browser: str | None = None
    proxy: str | None = None
    retry_sleep: float = 2.0
    max_attempts: int = 4

    # Keep this configurable because YouTube extractor clients
    # can change over time.
    player_clients: tuple[str, ...] = (
        "android",
        "web",
    )

    @staticmethod
    def _detect_cookie(cookie_file: str | None = None) -> Path | None:
        cookie_path = None
        if cookie_file is not None:
            cookie_path = Path(cookie_file)
            if not cookie_path.expanduser().is_file():
                logger.warning("Configured cookie file does not exist: %s", cookie_path,)
                cookie_path = None
            else:
                logger.info("Using configured cookie file: %s", cookie_path)
        return cookie_path
    
    @staticmethod
    def _detect_browser() -> str | None:
        browsers = {
            "chrome": ["google-chrome", "google-chrome-stable", "chrome"],
            "chromium": ["chromium", "chromium-browser"],
            "firefox": ["firefox"],
            "edge": ["microsoft-edge", "microsoft-edge-stable"],
            "brave": ["brave-browser", "brave"],
            "opera": ["opera"],
            "vivaldi": ["vivaldi"],
        }
        for browser, executables in browsers.items():
            for executable in executables:
                if shutil.which(executable):
                    logger.debug("Possible browser detected: %s",browser,)
                    return browser
        return None
            
    @classmethod
    def detect(cls, cookie_file: str | None = None, max_attempts: int = 4) -> YTDLPEnvironment:
        """ Automatically detect authentication and device environment settings
        """
        return cls(
            cookie_file=cls._detect_cookie(cookie_file),
            browser=cls._detect_browser(),
            max_attempts=max_attempts
        )

    @property
    def authentication_method(self) -> str:
        if self.cookie_file: return "cookie-file"
        if self.browser: return f"browser:{self.browser}"
        return "anonymous"

class YTDLPManager:
    FORMAT = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    def __init__(self, environment: YTDLPEnvironment):
        self.environment = environment

    def build_extractor_args(self) -> dict:
        """
            Build current YouTube extractor configuration.
    
            Important:
            - Do not hard-code android + web.
            - Let the PO-token provider handle current token generation.
            - mweb is the preferred client for current PO-token provider setups.
        """
        youtube_args = {"player_client": list(self.environment.player_clients),}
        # pot here
        return {"youtube": youtube_args}

    def build_options(
        self, output_template: str, progress_hook = None, *, simulate: bool = False
    ) -> dict:
        opts = {
            'format': self.FORMAT,
            'merge_output_format': 'mp4',
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "continuedl": True,
            "extractor_args": self.build_extractor_args(),
            "simulate": simulate,
            # Keep yt-dlp cache enabled.
            "cachedir": str(Path.home() / ".cache" / "yt-dlp"),
        }
        if progress_hook:
            opts["progress_hooks"] = [progress_hook,]
        if (cookie_path:=self.environment.cookie_file):
            opts["cookiefile"] = str(cookie_path)
            logger.debug("YDL Auth: Cookie File")
        elif (browser_spec:=self.environment.browser):
            # yt-dlp Python API expects:
            #   (browser, profile, keyring, container, ...)
            # depending on version.
            parts = browser_spec.split(":", 1)
            browser = parts[0]
            profile = parts[1] if len(parts) > 1 else None
            opts["cookiesfrombrowser"] = (browser, profile, None, None,)
            logger.debug("YDL Auth: browser cookies (%s)", browser_spec,)
        if (proxy:=self.environment.proxy):
            opts["proxy"] = proxy
        return opts

    @staticmethod
    def classify_error(error: Exception) -> str:
        msg = str(error).lower()
        patterns = {
            "AUTH": (
                "sign in to confirm",
                "login required",
                "authentication",
                "cookies",
                "cookies are no longer valid",
                "confirm you're not a bot",
            ),
            "PO_TOKEN": (
                "po token",
                "potoken",
                "proof of origin",
                "botguard",
                "attestation",
            ),
            "HTTP_403": (
                "http error 403",
                "403 forbidden",
                "forbidden",
            ),
            "RATE_LIMIT": (
                "http error 429",
                "too many requests",
                "rate limit",
            ),
            "FORMAT": (
                "requested format is not available",
                "requested format",
            ),
            "UNAVAILABLE": (
                "private video",
                "video unavailable",
                "video is unavailable",
                "has been removed",
                "deleted video",
            ),
            "GEO_BLOCKED": (
                "geo-restricted",
                "not available in your country",
                "country",
            ),
            "NETWORK": (
                "network error",
                "timed out",
                "timeout",
                "connection reset",
                "temporary failure",
            ),
        }
        for category, needles in patterns.items():
            if any(needle in msg for needle in needles):
                return category
        return "UNKNOWN"

    def extract_info(self, url: str) -> dict:
        """ Youtube Metadata Getter """
        options = self.build_options(
            "%(title)s.%(ext)s",
            simulate=True,
        )
        with env.yt_dlp.YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=False)

    def download(self, url: str, output_template: str, progress_hook=None) -> Path:
        options = self.build_options(
            output_template,
            progress_hook,
        )
        with env.yt_dlp.YoutubeDL(options) as ydl:
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
        raise FileNotFoundError(
            "yt-dlp completed without "
            f"producing an output file: "
            f"{requested}"
        )


YOUTUBE_NOISE_RE = re.compile(
    r"""(?ix)
    \b(
        official
        | official\s+audio
        | official\s+video
        | official\s+music\s+video
        | music\s+video
        | lyrics?
        | lyric\s+video
        | lirik
        | video
        | audio
        | visualizer
        | performance
        | live
        | live\s+performance
        | mv
        | hd
        | 4k
        | 8k
        | remastered
        | remaster
        | karaoke
        | instrumental
        | cover
        | sped\s*up
        | slowed\s*(?:\+\s*reverb)?
        | nightcore
    )\b
    """,
)

@dataclass
class LyricsCandidate:
    id: int | None
    title: str
    artist: str
    album: str
    duration: float
    plain_lyrics: str | None
    synced_lyrics: str | None
    
    @classmethod
    def from_api(cls, data: dict) -> LyricsCandidate:
        return cls(id=data.get("id"),
                   title=data.get("trackName") or "",
                   artist=data.get("artistName") or "",
                   album=data.get("albumName") or "",
                   duration=float(data.get("duration") or 0),
                   plain_lyrics=data.get("plainLyrics"),
                   synced_lyrics=data.get("syncedLyrics"),
        )

    @property
    def lyrics(self) -> str | None:
        return (self.plain_lyrics or self.synced_lyrics)


class LyricsFetcher:
    API = "https://lrclib.net/api" # LRC LIB API
    
    def __init__(self, session: requests.Session, **kwargs):
        self.session = session

    def _request(self, endpoint: str, params: dict) -> ...:
        url = f"{self.API}/{endpoint}"
        logger.debug("LRCLIB request: %s params=%r", endpoint, params,)
        response = self.session.get(url, params=params, timeout=30)
        if response.status_code == 429: # Rate Limit
            retry_after = (response.headers.get("Retry-After"))
            logger.warning("LRCLIB rate limit. Retry-After=%s", retry_after,)
            return None
        if response.status_code == 404: # Api Not Found
            return None
        if response.status_code != 200: # Others
            logger.warning("LRCLIB HTTP %s: %s", response.status_code, response.text[:300])
            return None
        try:
            return response.json()
        except ValueError:
            logger.warning("LRCLIB returned invalid JSON")
            return None

    @staticmethod
    def normalize_youtube_title(title: str) -> str:
        title = title.strip()
        # Remove translations such as:
        # Song Name // English Translation
        title = re.sub(r"\s*(?:///|//|｜|\|)\s*.*$", "", title)
        # Remove bracketed metadata.
        title = re.sub(r"\([^)]*\)", " ", title)
        title = re.sub(r"\[[^\]]*\]", " ", title)
        # Remove known YouTube metadata.
        title = YOUTUBE_NOISE_RE.sub(" ", title)
        title = re.sub(r"\s+", " ", title)
        return title.strip(" -–—")
        
    def parse_metadata(self, youtube_title: str, youtube_artist: str | None) -> tuple[str, str]:
        cleaned = self.normalize_youtube_title(youtube_title)
        artist = (youtube_artist or "").strip()
        title = cleaned
        match = re.match(
            r"^(?P<ArtistOrTitle>.+?)\s*[-–—]\s*(?P<TitleOrArtist>.+?)$",
            cleaned, re.VERBOSE,
        )
        if match:
            ext_artist_or_title = match.group("ArtistOrTitle").strip()
            ext_title_or_artist = match.group("TitleOrArtist").strip()
            if ext_artist_or_title and not ext_title_or_artist:
                # First part exist, it must be title
                title = ext_artist_or_title
            if ext_title_or_artist and not ext_artist_or_title:
                # Only 1 single part again, must be title
                title = ext_title_or_artist
            if ext_artist_or_title and ext_title_or_artist:
                if ext_artist_or_title.lower() in artist.lower():
                    artist = ext_artist_or_title
                    title = ext_title_or_artist
                elif ext_title_or_artist.lower() in artist.lower():
                    artist = ext_title_or_artist
                    title = ext_artist_or_title
        return title, artist

    def _get_exact(self, title: str, artist: str, duration: float) -> LyricsCandidate | None:
        if not title or not artist: return None
        data = self._request("get", {
            "track_name": title, "artist_name": artist,
            "duration": int(round(duration)),
        })
        if not isinstance(data, dict): return None
        candidate = LyricsCandidate.from_api(data)
        if not candidate.lyrics: return None
        return candidate

    def _search(self, params: dict) -> list[LyricsCandidate]:
        data = self._request("search", params)
        if not isinstance(data, list): return []
        return [
            LyricsCandidate.from_api(item)
            for item in data
            if isinstance(item, dict)
        ]


    def _build_searches(self, title: str, artist: str) -> list[dict]:
        searches: list[dict] = []
        # Highest precision.
        if title and artist:
            searches.append({
                "track_name": title,
                "artist_name": artist,
            })
        # Title only.
        if title: searches.append({"track_name": title})
        # General search.
        if artist and title: searches.append({"q": f"{artist} {title}"})
        # Last resort.
        if title: searches.append({"q": title})
        return searches
        
    def fetch_lyrics(self, youtube_title: str, youtube_artist: str, duration: float) -> str | None:
        title, artist = self.parse_metadata(youtube_title, youtube_artist)
        logger.info("Lyrics metadata: title=%r artist=%r duration=%ss",
            title, artist, duration
        )
        # 1. Exact LRCLIB lookup
        with rich.console.status("Trying exact LRCLIB match..."):
            exact = self._get_exact(title, artist, duration)
        if exact:
            logger.info("Exact LRCLIB match: %s - %s",
                exact.artist, exact.title,
            )
            return exact.lyrics
        # 2. Search
        candidates_by_id: dict[int | str, LyricsCandidate] = {}
        for params in self._build_searches(title, artist):
            logger.debug("LRCLIB search: %r", params)
            candidates = self._search(params)
            for candidate in candidates:
                key = (
                    candidate.id
                    if candidate.id is not None
                    else (
                        candidate.artist,
                        candidate.title,
                    )
                )
                candidates_by_id[key] = candidate
            if candidates: time.sleep(0.25) # add a bit of delay
        candidates = list(candidates_by_id.values())
        if not candidates:
            logger.warning("LRCLIB returned no candidates for %r - %r",
                artist, title
            )
            return None
        # 3. Score
        for candidate in candidates:
            self._score(candidate, title, artist, duration,)
        candidates.sort(key=lambda item: item.total_score, reverse=True)
        self._display_candidates(candidates)
        best = candidates[0]
        logger.info("Best LRCLIB candidate: %r - %r score=%.2f",
            best.artist, best.title, best.total_score,
        )
        # 4. Confidence gate
        if best.total_score < 70:
            logger.warning(
                "Best lyrics candidate is below "
                "confidence threshold: %.2f",
                best.total_score,
            )
            return None
        return best.lyrics



class SongDownloader:
    """ Youtube Aware downloader
    """
    def __init__(self, environment: YTDLPEnvironment | None = None,
                 *,
                 cookie_file: str | None = None,
                 max_attempts: int = 4,
                 **kwargs):
        self.session = env.requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) '
                'AppleWebKit/537.36 '
                '(KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })
        self.environment = environment or YTDLPEnvironment.detect(
            cookie_file, max_attempts
        )
        self.ytdlp = YTDLPManager(self.environment)
        self.lyrics_api = LyricsFetcher(session=self.session)
        self.without_lyrics = kwargs.pop("without_lyrics", False)
        if kwargs:
            logger.debug("Unused SongDownloader Kwargs %s", (kwargs,))

    @staticmethod
    def _progress_hook(progress: Progress, task_id: TaskID, data: dict) -> None:
        """ YTDLP Explicit Progress Hooks"""
        status = data.get("status")
        if status == "downloading":
            total = (
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
                or 0
            )
            downloaded = (
                data.get("downloaded_bytes")
                or 0
            )
            if total:
                progress.update(
                    task_id,
                    total=total,
                    completed=downloaded,
                    description="Downloading",
                )
        elif status == "finished":
            total = (
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
                or 0
            )
            progress.update(
                task_id,
                total=total or 1,
                completed=total or 1,
                description="Processing / merging",
            )

    def _retry_delay(self, attempt: int, failure_type: str) -> float:
        if failure_type == "RATE_LIMIT":
            return min(
                60,
                self.environment.retry_sleep
                * (2 ** (attempt - 1)),
            )
        if failure_type == "NETWORK":
            return min(
                30,
                self.environment.retry_sleep
                * (2 ** (attempt - 1)),
            )
        return self.environment.retry_sleep
    
    @staticmethod
    def _build_output_template(output_path: str | None, external_id: int | None,) -> str:
        filename = (
            f"{external_id:04d}_%(title)s.%(ext)s"
            if external_id is not None
            else "%(title)s.%(ext)s"
        )
        if not output_path:
            return str(Path(tempfile.gettempdir()) / filename)
        path = Path(output_path).expanduser()
        # Treat an existing extension as a file.
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        path.mkdir(parents=True, exist_ok=True,)
        return str(path / filename)

        
    def download(self, url: str, output_path: str | None = None, external_id: int | None = None, **kwargs) -> SimpleNamespace:
        output_template = self._build_output_template(output_path, external_id)
        last_error: Exception | None = None
        with rich.make_progress() as progress:
            overall = progress.add_task(
                "Preparing",
                total=3
                if not self.without_lyrics
                else 2,
            )
            download_task = progress.add_task(
                "Waiting for download...",
                total=None, visible=False,
            )
            for attempt in range(1, self.environment.max_attempts + 1):
                try:
                    logger.info("yt-dlp attempt %d/%d: %s",
                        attempt, self.environment.max_attempts, url
                    )
                    # METADATA
                    progress.update(overall, description=f"Extracting metadata (attempt {attempt})")
                    info = self.ytdlp.extract_info(url)
                    progress.update(overall, advance=1)
                    title = info.get("title", "unknown")
                    artist = (
                        info.get("artist")
                        or info.get("uploader", "Unknown")
                    )
                    duration = float(
                        info.get("duration", 0)
                        or 0
                    )
                    logger.info("YouTube metadata: title=%r artist=%r duration=%ss",
                        title, artist, duration,
                    )
                    # LYRICS
                    lyrics = None
                    if not self.without_lyrics:
                        progress.update(overall, description="Finding lyrics")
                        lyrics = self.lyrics_api.fetch_lyrics(title, artist, duration)
                        if not lyrics:
                            raise LyricsError("Lyrics not found with sufficient confidence.")
                        progress.update(overall, advance=1)
                    # DOWNLOAD
                    progress.update(overall, description="Downloading")
                    progress.update(
                        download_task,
                        visible=True,
                        description="Downloading",
                    )
                    def hook(data):
                        self._progress_hook(
                            progress,
                            download_task,
                            data,
                        )
                    filename = self.ytdlp.download(url, output_template, progress_hook=hook)
                    progress.update(
                        overall,
                        advance=1,
                        description="Complete",
                    )
                    # FINAL DISPLAY
                    self._show_result(
                        title=title, artist=artist,
                        duration=duration, filename=filename,
                        lyrics=lyrics,
                    )
                    return SimpleNamespace(
                        title=title, artist=artist,
                        duration=duration,
                        lyrics=lyrics, filename=filename,
                    )
                except LyricsError:
                    raise # Lyrics error need to be raise
                except Exception as exc:
                    last_error = exc
                    failure_type = self.ytdlp.classify_error(exc)
                    logger.warning("yt-dlp failure: type=%s error=%s",
                        failure_type, exc, exc_info=True,
                    )
                    if failure_type in {"UNAVAILABLE", "GEO_BLOCKED", "UNKNOWN"}:
                        raise YTDLPError(f"{failure_type}: {exc}") from exc
                    if attempt >= self.environment.max_attempts:cbreak
                    delay = self._retry_delay(attempt, failure_type)
                    progress.update(
                        overall,
                        description=(
                            f"{failure_type} — "
                            f"retrying in "
                            f"{delay:.1f}s"
                        ),
                    )
                    time.sleep(delay)
        raise YTDLPError(
            f"yt-dlp failed after "
            f"{self.environment.max_attempts} "
            f"attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _show_result(
        *,
        title: str,
        artist: str,
        duration: float,
        filename: Path,
        lyrics: str | None,
    ) -> None:
        table = rich.Table.grid(expand=True, padding=(0, 1))
        table.add_row("Artist", artist)
        table.add_row("Title", title)
        table.add_row("Duration", f"{duration:.0f}s")
        table.add_row("Lyrics", "Found" if lyrics else "Skipped")
        table.add_row("File", str(filename))
        rich.console.print()
        rich.console.print(rich.Panel(table, padding=1, title="Karaoke+ Download Complete", border_style="color(14)"))
        rich.console.print()



def get_track_file(inputpath: str, without_lyrics: bool) -> SimpleNamespace:
    inputpath = str(inputpath)
    parsed = urlparse(inputpath)
    if parsed.scheme in {"http", "https"}:
        logger.info("Downloading track source: %s", inputpath)
        return SongDownloader(without_lyrics=without_lyrics).download(inputpath)
    path = Path(inputpath)
    if not path.is_file():
        raise RuntimeError("Input file has not been downloaded and is not a file.")
    return SimpleNamespace(
        filename=str(path),
        lyrics=None,
    )
    