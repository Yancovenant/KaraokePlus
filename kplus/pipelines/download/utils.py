from enum import Enum

__all__ = [
    "DownloadError",
    "ErrorType",
    "LyricsError",
]

class LyricsError(Exception):
    """Raised when lyrics cannot be resolved with sufficient confidence."""


class DownloadError(Exception):
    """Raised when yt-dlp cannot complete the requested operation."""


class ErrorType(Enum):
    """ Base Exception for Network (YTDLP) Error """
    AUTH = (
        "sign in to confirm",
        "login required",
        "authentication",
        "cookies",
        "cookies are no longer valid",
        "confirm you're not a bot",
    )
    PO_TOKEN = (
        "po token",
        "potoken",
        "proof of origin",
        "botguard",
        "attestation",
    )
    HTTP_403 = (
        "http error 403",
        "403 forbidden",
        "forbidden",
    )
    RATE_LIMIT = (
        "http error 429",
        "too many requests",
        "rate limit",
    )
    FORMAT = (
        "requested format is not available",
        "requested format",
    )
    UNAVAILABLE = (
        "private video",
        "video unavailable",
        "video is unavailable",
        "has been removed",
        "deleted video",
    )
    GEO_BLOCKED = (
        "geo-restricted",
        "not available in your country",
        "country",
    )
    NETWORK = (
        "network error",
        "timed out",
        "timeout",
        "connection reset",
        "temporary failure",
    )
    UNKNOWN = ()

    @classmethod
    def _missing_(cls, exc: Exception):
        msg = str(exc).lower()
        for member in cls:
            if member == cls.UNKNOWN: continue
            if any(needle in msg for needle in member.value):
                return member
        return cls.UNKNOWN
