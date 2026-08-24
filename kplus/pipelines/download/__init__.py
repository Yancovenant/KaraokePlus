
from .downloader import Downloader, DownloadResult

__all__ = [
    "Downloader",
    "download_song"
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