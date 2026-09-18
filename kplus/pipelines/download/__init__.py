
from .downloader import Downloader, DownloadResult

__all__ = [
    "DownloadResult",
    "download_song",
    "extract_info",
    "extract_lyrics",
]

def download_song(
    url: str,
    output: str | None = None,
    external_id: int | None = None,
    *,
    no_lyrics: bool = False,
    **kwargs,
) -> DownloadResult:
    return (
        Downloader(**kwargs)
        .download(url, output, external_id, no_lyrics=no_lyrics)
    )

def extract_info(url: str, **kwargs) -> tuple[str, str, float]:
    return (
        Downloader(**kwargs)
        ._extract_info(url)
    )

def extract_lyrics(title: str, artist: str, duration: float, **kwargs) -> str:
    return (
        Downloader(**kwargs)
        .get_lyrics(title, artist, duration)
    )