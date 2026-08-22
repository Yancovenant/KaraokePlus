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
        browsers = ["chrome", "chromium", "firefox", "edge",
                    "brave", "opera", "vivaldi",]
        for browser in browsers:
            executable_names = {
                "chrome": ["google-chrome", "google-chrome-stable", "chrome"],
                "chromium": ["chromium", "chromium-browser"],
                "firefox": ["firefox"],
                "edge": ["microsoft-edge", "microsoft-edge-stable"],
                "brave": ["brave-browser", "brave"],
                "opera": ["opera"],
                "vivaldi": ["vivaldi"],
            }[browser]
            for executable in executable_names:
                if shutil.which(executable):
                    logger.debug("Possible browser detected: %s",browser,)
                    return browser
        return None
            
    @classmethod
    def detect(cls, cookie_file: str | None = None, max_attempts: int = 4) -> YTDLPEnvironment:
        """ Automatically detect authentication and device environment settings
        """
        cookie_path = cls._detect_cookie(cookie_file)
        browser = cls._detect_browser()
        return cls(
            cookie_file=cookie_path,
            browser=browser,
            max_attempts=max_attempts
        )


class SongDownloader:
    """ Youtube Aware downloader
    """
    def __init__(self, **kwargs):
        self.session = env.requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) '
                'AppleWebKit/537.36 '
                '(KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })
        self.YDLEnvironment: YTDLPEnvironment = kwargs.pop("YDLEnvironment", YTDLPEnvironment.detect(kwargs.pop("cookie_file", None), kwargs.pop("max_attempts", 4)))
        self.without_lyrics = kwargs.pop("without_lyrics", False)
        if kwargs:
            logger.debug("Song Downloader Unused Kwargs %s", (kwargs,))

    def _build_extractor_args(self) -> dict:
        """
            Build current YouTube extractor configuration.
    
            Important:
            - Do not hard-code android + web.
            - Let the PO-token provider handle current token generation.
            - mweb is the preferred client for current PO-token provider setups.
        """
        youtube_args = {"player_client": ["android", "web"],}
        # pot here
        return {"youtube": youtube_args}

    def _build_ydl_opts(self, output_template: str, progress_hook, *, simulate: bool = False) -> dict:
        opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "continuedl": True,
            "progress_hooks": [progress_hook,],
            "extractor_args": self._build_extractor_args(),
            "simulate": simulate,
            # Keep yt-dlp cache enabled.
            "cachedir": str(Path.home() / ".cache" / "yt-dlp"),
        }
        if (cookie_path:=self.YDLEnvironment.cookie_file):
            opts["cookiefile"] = str(cookie_path)
            logger.debug("YDL Auth: Cookie File")
        elif (browser_spec:=self.YDLEnvironment.browser):
            # yt-dlp Python API expects:
            #   (browser, profile, keyring, container, ...)
            # depending on version.
            parts = browser_spec.split(":", 1)
            browser = parts[0]
            profile = parts[1] if len(parts) > 1 else None
            opts["cookiesfrombrowser"] = (browser, profile, None, None,)
            logger.debug("YDL Auth: browser cookies (%s)", browser_spec,)
        if (proxy:=self.YDLEnvironment.proxy):
            opts["proxy"] = proxy
        return opts

    @staticmethod
    def _classify_error(error: Exception) -> str:
        msg = str(error).lower()
        if any(x in msg for x in (
            "sign in to confirm",
            "login required",
            "authentication",
            "cookies",
            "cookies are no longer valid",
            "confirm you're not a bot",
        )): return "AUTH"
        if any(x in msg for x in (
            "po token",
            "potoken",
            "proof of origin",
            "botguard",
            "attestation",
        )): return "PO_TOKEN"
        if any(x in msg for x in (
            "http error 403",
            "403 forbidden",
            "forbidden",
        )): return "HTTP_403"
        if any(x in msg for x in (
            "http error 429",
            "too many requests",
            "rate limit",
        )): return "RATE_LIMIT"
        if any(x in msg for x in (
            "private video",
            "video unavailable",
            "video is unavailable",
            "has been removed",
            "deleted video",
        )): return "UNAVAILABLE"
        if any(x in msg for x in (
            "geo-restricted",
            "not available in your country",
            "country",
        )): return "GEO_BLOCKED"
        if any(x in msg for x in (
            "network error",
            "timed out",
            "timeout",
            "connection reset",
            "temporary failure",
        )): return "NETWORK"
        return "UNKNOWN"

    def download(self, url: str, output_path: str | None = None, external_id: int | None = None, **kwargs) -> SimpleNamespace:
        filename_template = (
            f"{external_id:04d}_%(title)s.%(ext)s"
            if external_id is not None
            else "%(title)s.%(ext)s"
        )
        if not output_path: output_dir = Path(tempfile.gettempdir())
        else: output_dir = Path(output_path).expanduser()
        if output_dir.suffix == "":
            output_dir.mkdir(parents=True, exist_ok=True)
            output_template = str(output_dir / filename_template)
        else:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            output_template = str(output_dir)
        last_error = None
        for attempt in range(1, self.YDLEnvironment.max_attempts + 1,):
            logger.info("yt-dlp attempt %d/%d: %s", attempt, self.YDLEnvironment.max_attempts, url,)
            try:
                with MainProgress(total=2 if self.without_lyrics else 3, desc=f"Downloading {url}", unit="step") as main_bar:
                    main_bar.pbar.set_description("Extracting Info")
                    info_opts = self._build_ydl_opts(output_template, SubProgress(), simulate=True)
                    with env.yt_dlp.YoutubeDL(info_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                    main_bar.update(1)
                    title = info.get("title", "unknown")
                    artist = info.get("artist") or info.get("uploader", "Unknown")
                    duration = info.get("duration", 0.0)
                    logger.debug("Information: "
                                    "title=%r artist=%r duration=%ss",
                                    title, artist, duration,)
                    lyrics = None
                    if not self.without_lyrics:
                        main_bar.pbar.set_description("Fetching Lyrics")
                        if not (lyrics := LyricsFetcher(session=self.session).fetch_lyrics(title, artist, duration)):
                            raise LyricsError("!!! Lyrics not found, cannot continue")
                        main_bar.update(1)
                    main_bar.pbar.set_description("Downloading..")
                    download_opts = self._build_ydl_opts(output_template, SubProgress(),)
                    with env.yt_dlp.YoutubeDL(download_opts) as ydl:
                        filename = Path(ydl.prepare_filename(info))
                        if not filename.exists():
                            ydl.download([url])
                    if not filename.exists():
                        raise FileNotFoundError(
                            f"yt-dlp reported success but "
                            f"output file does not exist: "
                            f"{filename}"
                        )
                    main_bar.update(1)
                    logger.info("Download successful: %s", filename,)
                    return SimpleNamespace(title=title, artist=artist, duration=duration, lyrics=lyrics, filename=filename)
            except LyricsError:
                # Lyrics error not ytdlp problem
                raise
            except Exception as exc:
                last_error = exc
                failure_type = self._classify_error(exc)
                logger.warning("yt-dlp download attempt failed "
                            "(type=%s): %s",
                            failure_type, exc, exc_info=True,)
                # Permanent failure.
                if failure_type in {"UNAVAILABLE","GEO_BLOCKED",}:
                    raise RuntimeError(
                        f"Unable to download video: "
                        f"{failure_type}: {exc}"
                    ) from exc
                if attempt >= self.YDLEnvironment.max_attempts:
                    break
                if failure_type == "AUTH":
                    logger.warning("Authentication failure detected. Rebuilding yt-dlp authentication configuration.")
                    # If we have a cookie file, do not blindly delete it.
                    # Re-reading it on the next attempt is enough.
                if failure_type == "PO_TOKEN":
                    logger.warning("PO-token related failure. Forcing a fresh yt-dlp extraction attempt.")
                if failure_type == "HTTP_403":
                    logger.warning("HTTP 403 detected. Retrying through the recovery strategy.")
                if failure_type == "RATE_LIMIT":
                    delay = min(60, self.YDLEnvironment.retry_sleep * (2 ** max(0, attempt - 1)),)
                    logger.warning("Rate limit detected. Sleeping %.1f seconds.", delay,)
                    time.sleep(delay)
                if failure_type == "NETWORK":
                    delay = min(30, self.YDLEnvironment.retry_sleep * (2 ** max(0, attempt - 1)),)
                    logger.warning("Network failure. Sleeping %.1f seconds.", delay,)
                    time.sleep(delay)

                time.sleep(self.YDLEnvironment.retry_sleep)
        raise RuntimeError(
            f"yt-dlp failed after "
            f"{self.YDLEnvironment.max_attempts} attempts: "
            f"{last_error}"
        ) from last_error


class LyricsError(Exception):
    """ Return error if lyrics not found
    """


YOUTUBE_TITLE_NOISE = re.compile(
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

class LyricsFetcher:
    def __init__(self, **kwargs):
        self.session = kwargs.pop("session")
        
    def _fetch_lyrics_api(self, endpoint: str, params: dict) -> str | None:
        try:
            res = self.session.get(f"https://lrclib.net/api/{endpoint}", params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                from rich import inspect
                #inspect(data)
        except Exception as e:
            logger.warning(f"Lyrics API error ({endpoint}): {e}")
            raise
        return None
        
    def fetch_lyrics(self, title: str, artist: str, duration: float) -> str:
        title = title.strip()
        title = YOUTUBE_TITLE_NOISE.sub("", title)
        title = re.sub(r"\([^)]*\)", " ", title)
        title = re.sub(r"\[[^\]]*\]", " ", title)
        # Remove common YouTube translation separators.
        title = re.sub(r"\s*(//|///|\||｜)\s*.*$", "", title)
        # Normalize Whitespace
        title = re.sub(r"\s+", " ", title)
        match = re.match(r"^\s*(?P<artist>.+?)\s+[-–—]\s+(?P<title>.+?)\s*$",
                         title,)
        if match:
            artist_part = match.group("artist").strip()
            title_part = match.group("title").strip()
            artist = artist_part if artist_part else artist
            title = title_part if title_part else title
        logger.debug(f">> Getting Lyrics for {title} - {artist} ({duration})")
        to_searchs = [
            {"track_name": title, "artist_name": artist},
            {"track_name": title},
            {"q": f"{artist} {title}"},
            {"q": title},
        ]
        for param in to_searchs:
            if l:=self._fetch_lyrics_api("search", param): return l

class SongDownloader2:
    def __init__(self, without_lyrics: bool = False):
        self.session = env.requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
        self.opts = {
            'cookiefile': 'cookies.txt',
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'extractor_args': {'youtube': {
                'player_client': ['android', 'web'],
                'skip': ['webpage']}},
            'no_warnings': True, 'quiet': True,
            'progress_hooks': [SubProgress()]}
        self.without_lyrics = without_lyrics

    def _fetch_lyrics_api(self, endpoint: str, params: dict) -> str | None:
        try:
            res = self.session.get(f"https://lrclib.net/api/{endpoint}", params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                logger.debug(f"Receive response: {type(data)} | {data}")
                if isinstance(data, list):
                    if len(data) > 0:
                        return data[0].get("plainLyrics")
                else: return data.get("plainLyrics")
        except Exception as e:
            logger.warning(f"Lyrics API error ({endpoint}): {e}")
            raise
        return None

    def fetch_lyrics(self, title: str, artist: str, duration: int) -> str | None:
        if ' - ' in title:
            parts = title.split(' - ', 1)
            if parts[0].strip(): artist = parts[0].strip()
            title = parts[1].strip()
        clean_title = re.sub(r'\(.*?\)|\[.*?\]|official.*|music.*|video.*|lyrics.*|lirik.*|mv.*|hd|4k|[^\w\s]', '', title, flags=re.IGNORECASE).strip()
        logger.debug(f">> Getting Lyrics for {clean_title} - {artist} ({duration})")
        if artist and clean_title:
            if l := self._fetch_lyrics_api("get", {"artist_name": artist, "track_name": clean_title, "duration": int(duration)}): return l
            if l := self._fetch_lyrics_api("get", {"artist_name": artist, "track_name": clean_title}): return l
        query = f"{artist} {clean_title}" if artist else clean_title
        if l := self._fetch_lyrics_api("search", {"q": query}): return l
        return None

    def download(self, url: str, output_path: str | None = None, external_id: int | None = None) -> tuple[str, str, str, str | None, Path] | None:
        filepath = f"{external_id:04d}_%(title)s.%(ext)s" if external_id is not None else "%(title)s.%(ext)s"
        if not output_path: output_path = tempfile.gettempdir() + filepath
        if os.path.exists(output_path): logger.debug(f"File already exist..., {output_path}"); return output_path
        self.opts.update({'outtmpl': filepath,})
        task_total = 2 if self.without_lyrics else 3
        with env.yt_dlp.YoutubeDL(self.opts) as ydl:
            with MainProgress(total=task_total, desc=f"Downloading {url}", unit="step") as main_bar:
                main_bar.pbar.set_description("Extracting Info")
                info = ydl.extract_info(url, download=False)
                main_bar.update(1)
                title = info.get("title", "unknown")
                artist = info.get("artist") or info.get("uploader", "Unknown")
                duration = info.get("duration", 0.0)
                logger.debug("Information: title: `%s`, artist: `%s`, duration: `%ds`" % (title, artist, duration))  # noqa: UP031
                lyrics = None
                if not self.without_lyrics:
                    main_bar.pbar.set_description("Fetching Lyrics")
                    if not (lyrics := self.fetch_lyrics(title, artist, duration)):
                        raise ValueError("!!! Lyrics not found, cannot continue")
                    main_bar.update(1)
                filename = Path(ydl.prepare_filename(info))
                main_bar.pbar.set_description("Downloading..")
                if not filename.exists():
                    ydl.download([url])
                main_bar.update(1)
                logger.debug("Information: lyrics: `{}...`, filename: `{}`".format(lyrics[:25] if lyrics else "-No Lyrics-", filename))
            return SimpleNamespace(title=title, artist=artist, duration=duration, lyrics=lyrics, filename=filename)


def get_track_file(inputpath: str, without_lyrics: bool) -> SimpleNamespace:
    inputpath = str(inputpath)
    info = None
    try:
        parsed = urlparse(inputpath)
        if parsed.scheme in ("http", "https"):
            logger.info(f"Downloading track source from URL: {inputpath}")
            info = SongDownloader(without_lyrics=without_lyrics).download(inputpath)
            return info
    except Exception as err:
        # is this a filepath?
        logger.warning("!!! Exception on parsing URL as an input: %s", str(err))
    if not Path(inputpath).is_file():
        raise RuntimeError("!!! input file have not been download or is not a file")
    return SimpleNamespace(filename=inputpath, lyrics=None)