
import logging
from pathlib import Path

from .asr import *
from .audio import detect_audio_activity
from .download import DownloadResult, download_song
from .lyrics import *
from .separate import separate_song

logger = logging.getLogger(__name__)

__all__ = [
    "align",
    "align2ref",
    "detect_audio_activity",
    "detect_language",
    "download_song",
    "ensure_file",
    "multi_align",
    "refine",
    "separate_song",
    "transcribe",
]
###########################################
# Ensuring videopath exists if URL download
###########################################
import json
from urllib.parse import urlparse

from kplus.tools import safepath, search_for_path


def ensure_file(inputpath: str | Path, *, no_lyrics: bool = False, **kwargs) -> DownloadResult:
    inputpath = str(inputpath)
    if urlparse(inputpath).scheme in {"https", "http"}:
        logger.info("Downloading track source %s", inputpath)
        return download_song(inputpath, no_lyrics=no_lyrics, **kwargs)
    path = Path(inputpath)
    if not path.is_file():
        raise RuntimeError("Input file has not been downloaded and is not a file.")
    # Else path Exist, get its data .json if exists
    title, artist, duration, lyrics = None, None, None, None
    dirpath = search_for_path(path.stem)
    if dirpath:
        filename = safepath(path.stem)
        datafile = dirpath / Path(filename + ".json")
        lyricfile = dirpath / Path(filename + ".txt")
        if datafile.is_file():
            try:
                with open(datafile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title, artist, duration = data.get("title"), data.get("artist"), float(data.get("duration"))
            except Exception as e:
                logger.warning("Could not read metadata for %s: %s", datafile, e)
        if lyricfile.is_file():
            try:
                with open(datafile, "r", encoding="utf-8") as f:
                    lyrics = f.read()
            except Exception as e:
                logger.warning("Couldn't read existing lyrics file at %s: %s", lyricfile, e)
    return DownloadResult(
        title=title,
        artist=artist,
        duration=duration,
        lyrics=lyrics,
        filepath=str(path),
    )
