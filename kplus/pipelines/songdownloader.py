import logging
import tempfile
import os
import re

from pathlib import Path
from typing import Dict, Optional
from types import SimpleNamespace
from urllib.parse import urlparse

from kplus.environment import env
from kplus.tools.progress import MainProgress, SubProgress


logger = logging.getLogger(__name__)


class SongDownloader:
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

    def _fetch_lyrics_api(self, endpoint: str, params: Dict) -> Optional[str]:
        try:
            res = self.session.get(f"https://lrclib.net/api/{endpoint}", params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and data: return data[0].get("plainLyrics")
                else: return data.get("plainLyrics")
        except Exception as e:
            logger.warning(f"Lyrics API error ({endpoint}): {e}")
        return None

    def fetch_lyrics(self, title: str, artist: str, duration: int) -> Optional[str]:
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

    def download(self, url: str, output_path: Optional[str] = None, external_id: Optional[int] = None) -> Optional[tuple[str, str, str, Optional[str], Path]]:
        filepath = f"{external_id:04d}_%(title)s.%(ext)s" if external_id is not None else "%(title)s.%(ext)s"
        if not output_path: output_path = tempfile.gettempdir() + filepath
        if os.path.exists(output_path): logger.debug(f"File already exist..., {output_path}"); return output_path
        self.opts.update({'outtmpl': filepath,})
        task_total = 2 if self.without_lyrics else 3
        with env.yt_dlp.YoutubeDL(self.opts) as ydl:
            with MainProgress(total=task_total, desc="Downloading %s" % url, unit="step") as main_bar:
                main_bar.pbar.set_description("Extracting Info")
                info = ydl.extract_info(url, download=False)
                main_bar.update(1)
                title = info.get("title", "unknown")
                artist = info.get("artist") or info.get("uploader", "Unknown")
                duration = info.get("duration", 0.0)
                logger.debug("Information: title: `%s`, artist: `%s`, duration: `%ds`" % (title, artist, duration))
                lyrics = None
                if not self.without_lyrics:
                    main_bar.pbar.set_description("Fetching Lyrics")
                    if not (lyrics := self.fetch_lyrics(title, artist, duration)):
                        raise Exception("!!! Lyrics not found, cannot continue")
                    main_bar.update(1)
                filename = Path(ydl.prepare_filename(info))
                main_bar.pbar.set_description("Downloading..")
                if not filename.exists():
                    ydl.download([url])
                main_bar.update(1)
                logger.debug("Information: lyrics: `%s...`, filename: `%s`" % (lyrics[:25] if lyrics else "-No Lyrics-", filename))
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
        pass
    return SimpleNamespace(filename=inputpath)