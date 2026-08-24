from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import requests
from requests.exceptions import HTTPError

from kplus import env
from kplus.tools import rich, similarity, token_similarity

logger = logging.getLogger(__name__)

__all__ = [
    "Lrclib"
]

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

@dataclass(slots=True)
class LyricsCandidate:
    id: int | None
    title: str
    artist: str
    album: str
    duration: float
    plain_lyrics: str | None
    synced_lyrics: str | None

    title_score: float = 0.0
    artist_score: float = 0.0
    duration_score: float = 0.0
    total_score: float = 0.0

    @classmethod
    def from_api(cls, data: dict) -> LyricsCandidate:
        return cls(
            id=data.get("id"),
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

    @staticmethod
    def _duration_score(src: float, other: float) -> float:
        """ Simply compare both duration gap """
        if not src or not other: return 0.0
        difference = abs(float(src) - float(other))
        if difference <= 2: return 1.0
        if difference <= 5: return 0.8
        if difference <= 10: return 0.5
        if difference <= 20: return 0.2
        return 0.0

    def make_score(self, title: str, artist: str, duration: float) -> None:
        self.title_score = similarity(title, self.title)
        self.artist_score = similarity(artist, self.artist)
        self.duration_score = self._duration_score(duration, self.duration)
        token_score = token_similarity(title, self.title)
        self.total_score = (
            self.title_score * 45 # Title most important
            + (token_score * 20)
            + (self.artist_score * 25) # Secondly is artist
            + (self.duration_score * 10)
        )

class LyricsFetcherMixin:
    name: str
    API: str
    def __init__(self, session: requests.Session, **kwargs):
        self.session = session

    def _make_request(self, endpoint: str, params: str) -> dict | None:
        url = f"{self.API}/{endpoint}"
        logger.debug("%s request: %s params=%r", self.name, endpoint, params,)
        try:
            res = self.session.get(url, params=params, timeout=30)
            res.raise_for_status()
            return res.json()
        except HTTPError as e:
            logger.warning("%s HTTPError %s: %s", self.name, e.response.status_code, e)
        except ValueError as e:
            logger.warning("%s returned invalid JSON", self.name)
        return None

    def _clean_title(self, title: str) -> str:
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
        
    def normalize(self, title: str, artist: str | None = None) -> tuple[str, str]:
        title, artist = self._clean_title(title), (artist or "").strip()
        match = re.match(
            r"^(?P<ArtistOrTitle>.+?)\s*[-–—]\s*(?P<TitleOrArtist>.+?)$",
            title, re.VERBOSE,
        )
        if match:
            ext_artist_or_title = match.group("ArtistOrTitle").strip()
            ext_title_or_artist = match.group("TitleOrArtist").strip()
            if ext_artist_or_title and not ext_title_or_artist:
                # First part exist, it must be title
                title = ext_artist_or_title
            elif ext_title_or_artist and not ext_artist_or_title:
                # Only 1 single part again, must be title
                title = ext_title_or_artist
            elif ext_artist_or_title and ext_title_or_artist:
                if (ext_artist_or_title.lower() in artist.lower()
                   or artist.lower() in ext_artist_or_title.lower()):
                    artist = ext_artist_or_title
                    title = ext_title_or_artist
                elif ext_title_or_artist.lower() in artist.lower():
                    artist = ext_title_or_artist
                    title = ext_artist_or_title
        return title, artist

    def resolve_lyric_candidates(self, title: str, artist: str, duration: float, cands: list[LyricsCandidate]) -> str | None:
        if not cands:
            logger.warning("%s returned no candidates for %r - %r", self.name, artist, title)
            return None
        for cand in cands: cand.make_score(title, artist, duration)
        cands.sort(key=lambda item: item.total_score, reverse=True)
        if env.verbose:
            self.print(cands)
        best = cands[0]
        logger.info("Best LRCLIB candidate: %r - %r score=%.2f", best.artist, best.title, best.total_score)
        if best.total_score < 70: # Gating Confidence
            logger.warning("Best lyrics candidate is below confidence threshold: %.2f", best.total_score,)
            return None
        return best.lyrics

    def print(self, cands: list[LyricsCandidate]) -> None:
        if not cands: return
        table = rich.Table(title="LRCLIB Candidates", show_lines=False)
        table.add_column("#", justify="right",)
        table.add_column("Score", justify="right")
        table.add_column("Artist",)
        table.add_column("Title")
        table.add_column("Duration", justify="right",)
        for index, candidate in enumerate(cands):
            table.add_row(
                str(index),
                f"{candidate.total_score:.1f}",
                candidate.artist,
                candidate.title,
                f"{candidate.duration:.0f}s",
            )
        logger.info(table)
        
    def get_lyrics(self):
        raise NotImplementedError()

class Lrclib(LyricsFetcherMixin):
    name = "LRCLIB"
    API = "https://lrclib.net/api"

    def _get(self, title: str, artist: str, duration: float) -> str | None:
        if not title or not artist: return None
        data = self._make_request("get", {
            "track_name": title, "artist_name": artist,
            "duration": round(duration),
        })
        if not isinstance(data, dict):
            return None
        candidate = LyricsCandidate.from_api(data)
        if not candidate.lyrics:
            return None
        return candidate.lyrics

    def _search(self, params: dict[str, ...]) -> list[LyricsCandidate]:
        data = self._make_request("search", params)
        if not isinstance(data, list): return []
        return [
            LyricsCandidate.from_api(item)
            for item in data
            if isinstance(item, dict)
        ]

    def searches(self, title: str, artist: str): # Iterable
        if title and artist:
            yield {"track_name": title, "artist_name": artist,}
            yield {"q": f"{artist} {title}"}
        if title:
            yield {"track_name": title}
            yield {"q": title}
            
    def get_lyrics(self,
        title: str | None = None,
        artist: str | None = None,
        duration: float | None = None,
    ) -> str | None:
        if not title:
            return None
        title, artist = self.normalize(title, artist)
        logger.info("Lyrics metadata:\n  title=%r\n  artist=%r\n  duration=%ss", title, artist, duration)
        with rich.console.status("LRCLIB Trying direct Get"):
            if (lyrics:=self._get(title, artist, duration)):
                logger.info("Found match lyrics")
                return lyrics
        with rich.console.status("LRCLIB Trying search candidates") as st:
            cans_by_id = {}
            for params in self.searches(title, artist):
                candidates = self._search(params)
                for can in candidates:
                    key = (
                        can.id if can.id is not None
                        else (can.artist, can.title)
                    )
                    cans_by_id[key] = can
                if candidates: time.sleep(0.25) # add a bit of delay
            return self.resolve_lyric_candidates(title, artist, duration, list(cans_by_id.values()))
        