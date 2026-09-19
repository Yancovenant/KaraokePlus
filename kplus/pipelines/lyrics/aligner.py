import logging
from collections import defaultdict

from kplus import env
from kplus.pipelines.utils import (
    ASRResult,
    AudioSegment,
    TextTiming,
    WordTiming,
    overlap,
)
from kplus.tools import rich

from .sequence_aligner import SequenceAligner
from .utils import Tokens

__all__ = [
    "LyricAligner"
]

logger = logging.getLogger(__name__)


class LyricAligner:
    """ Main For Lyrics Aligner """

    def __init__(
        self,
        *,
        raise_if_not_reliable: bool = True
    ):
        self.sequence_aligner = SequenceAligner(
            raise_if_not_reliable=raise_if_not_reliable
        )

    def asr2ref(
        self,
        hypothesis: ASRResult,
        reference: str,
        audiosegments: list[AudioSegment],
        *,
        no_show: bool = True,
        **kwargs
    ) -> tuple[ASRResult, list[AudioSegment]]:
        """
            ::param no_show will hide plot detection
        """
        ref_tokens, hyp_tokens = (
            self.sequence_aligner(
                Tokens.from_reference(reference),
                Tokens.from_asr(hypothesis.texts)
            )
        )
        ##
        # TODO: Create multiple test here
        ##
        from kplus.tools.text import similarity

        ref_by_audio = defaultdict(list)
        for i, audio_segment in enumerate(audiosegments):
            ref_by_audio[i] = []
            for j, token in enumerate(ref_tokens):
                if token.start is None:
                    continue
                if overlap(audio_segment, token):
                    ref_by_audio[i].append(j)

        # Interpolation
        for audio_id, token_ids in ref_by_audio.items():
            if token_ids:
                ref_by_audio[audio_id] = list(range(
                    min(token_ids), max(token_ids) + 1
                ))

        # Extrapolation
        unused_gaps = {}
        for i, audio_segment in enumerate(audiosegments):
            unused_gaps[i] = (None, None)
            token_ids = ref_by_audio[i]
            if not token_ids:
                continue
            start_gap = ref_tokens[min(token_ids)].start - audio_segment.start
            end_gap = audio_segment.end - ref_tokens[max(token_ids)].end
            unused_gaps[i] = (start_gap, end_gap)

        last_word_idx = -1
        last_audio_id = len(audiosegments) - 1

        def _check_missing(max_id: int):
            return list(range(last_word_idx + 1, max_id))

        for i in range(len(audiosegments)):
            last_audio_and_still_have_word_missing = (
                i == last_audio_id and
                (missing := _check_missing(len(ref_tokens)))
            )
            if last_audio_and_still_have_word_missing:
                logger.debug(
                    "Subject already in a last segment, "
                    f"but there is still missing word {missing}")
                prev_anchor_audio_id = max(
                    k for k, v in ref_by_audio.items()
                    if v
                )
                _, prev_end_gap = unused_gaps[prev_anchor_audio_id]
                if prev_end_gap > 0.4:
                    logger.debug(
                        "  Previous end gap is still large, "
                        "assigned it to previous "
                        f"{ref_by_audio[prev_anchor_audio_id]} | {missing}"
                    )
                    ref_by_audio[prev_anchor_audio_id] = sorted(
                        ref_by_audio[prev_anchor_audio_id] + missing
                    )
                else:
                    logger.debug(
                        "  Previous end gap insufficient, "
                        "try getting the next audio"
                    )
                    if prev_anchor_audio_id < i:
                        logger.debug(
                            "  Next audio exists, "
                            f"{prev_anchor_audio_id+1} | "
                            f"{ref_by_audio[prev_anchor_audio_id+1]}"
                        )
                        ref_by_audio[prev_anchor_audio_id+1] = sorted(missing)

            token_ids = ref_by_audio[i]
            if not token_ids:
                continue

            there_still_prev_word_being_skipped = (
                min(token_ids) > last_word_idx + 1 and
                (missing := _check_missing(min(token_ids)))
            )
            if there_still_prev_word_being_skipped:
                logger.debug(
                    "Subject still skipping words: "
                    f"{ref_by_audio[i-1]} | {missing} | "
                    f"{ref_by_audio[i]}"
                )
                missing_text = " ".join(ref_tokens[j].word for j in missing)
                known_text = " ".join(ref_tokens[j].word for j in token_ids)
                if prev_anchor_audio_id := next((
                    j for j in range(i - 1, -1, -1)
                    if ref_by_audio[j]
                ), None):
                    logger.debug(
                        "  Previous Anchor Audio Exists: "
                        f"{prev_anchor_audio_id}"
                    )
                    _, anchor_end_gap = unused_gaps[prev_anchor_audio_id]
                    if anchor_end_gap > 1.0:
                        logger.debug(
                            "  Previous anchor gap still high "
                            "assigned to previous anchor "
                            f"{sorted(
                                ref_by_audio[prev_anchor_audio_id] + missing
                            )}"
                        )
                        ref_by_audio[prev_anchor_audio_id] = sorted(
                            ref_by_audio[prev_anchor_audio_id] + missing
                        )
                    else:
                        logger.debug(
                            "  Previous anchor gap insufficient, "
                            "combining to previous audio: "
                            f"{ref_by_audio[i-1]} | {missing} | {token_ids}"
                        )
                        ref_by_audio[i] = sorted(
                            token_ids + missing + ref_by_audio[i-1]
                        )
                        ref_by_audio[i-1] = sorted(
                            token_ids + missing + ref_by_audio[i-1]
                        )
                elif (ratio := similarity(missing_text, known_text)) > 0.7:
                    logger.debug(
                        f"  Similarity high ({ratio}) between\n"
                        f"{missing_text} | {known_text}\n"
                        f"{sorted(token_ids + missing)}"
                    )
                    ref_by_audio[i] = sorted(token_ids + missing)
                else:
                    logger.debug(
                        "  Combining to previous Audio: "
                        f"{ref_by_audio[i-1]} | {missing} | {token_ids}"
                    )
                    ref_by_audio[i] = sorted(
                        token_ids + missing + ref_by_audio[i-1]
                    )
                    ref_by_audio[i-1] = sorted(
                        token_ids + missing + ref_by_audio[i-1]
                    )

            token_ids = ref_by_audio[i]

            start_gap, end_gap = unused_gaps[i]
            if i < last_audio_id and (next_token_ids := ref_by_audio[i+1]):
                next_start_gap, _ = (
                    unused_gaps[i+1]
                    if i < last_audio_id
                    else (None, None)
                )
                if missing := sorted(set(range(
                    max(token_ids) + 1, min(next_token_ids)
                ))):
                    logger.debug(
                        "Subject is missing: "
                        f"{token_ids} | {missing} | {next_token_ids}"
                    )
                    if next_start_gap > end_gap:
                        logger.debug(
                            "  Belong to the next: "
                            f"{sorted(
                                next_token_ids + missing
                            )}"
                        )
                        ref_by_audio[i+1] = sorted(next_token_ids + missing)
                    else:
                        logger.debug(
                            "  Combine Both: "
                            f"{sorted(
                                token_ids + missing + next_token_ids
                            )}"
                        )
                        ref_by_audio[i] = sorted(
                            token_ids + missing + next_token_ids
                        )
                elif end_gap > 2.0 or next_start_gap > 1.0:
                    # No Unknown but the gap in either each segment is so long
                    # So we combine
                    logger.debug(
                        "Subject for combine: "
                        f"{ref_by_audio[i]} | {ref_by_audio[i+1]}"
                    )
                    ref_by_audio[i] = sorted(
                        ref_by_audio[i] + ref_by_audio[i+1]
                    )
                    ref_by_audio[i+1] = sorted(
                        ref_by_audio[i] + ref_by_audio[i+1]
                    )

            token_ids = ref_by_audio[i]
            last_word_idx = max(token_ids)

        assigned = [
            token_id
            for token_ids in ref_by_audio.values()
            for token_id in token_ids
        ]
        missing = (
            set(range(len(ref_tokens))) - set(assigned)
        )
        if missing:
            logger.warning(f"Missing: {missing}")

        def should_merge(i, j) -> bool:
            a_tokens = ref_by_audio[i]
            b_tokens = ref_by_audio[j]
            if not a_tokens or not b_tokens:
                return False
            return bool(not set(a_tokens).isdisjoint(set(b_tokens)))

        # Clustering
        clusters, current = [], None
        for i in range(len(audiosegments)):
            if not ref_by_audio[i]:
                continue
            if current is None:
                current = {
                    "audio_ids": [i],
                    "token_ids": list(ref_by_audio[i])
                }
                continue
            prev_i = current["audio_ids"][-1]
            if should_merge(prev_i, i):
                current["audio_ids"].append(i)
                current["token_ids"].extend(ref_by_audio[i])
                current["token_ids"] = sorted(set(current["token_ids"]))
            else:
                clusters.append(current)
                current = {
                    "audio_ids": [i],
                    "token_ids": list(ref_by_audio[i])
                }
        if current:
            clusters.append(current)

        new_audiosegments = []
        new_texts = []
        for i, cluster in enumerate(clusters):
            audio_ids = cluster["audio_ids"]
            token_ids = cluster["token_ids"]
            first = audiosegments[audio_ids[0]]
            last = audiosegments[audio_ids[-1]]
            new_audiosegments.append(
                AudioSegment(
                    start=first.start, end=last.end,
                )
            )
            words, langs = [], []
            for token_id in token_ids:
                token = ref_tokens[token_id]
                langs.append(token.language)
                words.append(WordTiming(
                    start=token.start,
                    end=token.end,
                    score=token.score,
                    word=token.word,
                ))
            if not langs:
                langs = ["English"]
            new_texts.append(TextTiming(words=words, language=langs[0]))
        assert len(new_texts) == len(new_audiosegments)
        new_words = [w for text in new_texts for w in text.words]
        assert len(new_words) == len(ref_tokens)
        if env.verbose and not no_show:
            for i, (text, audio_segment) in enumerate(zip(
                new_texts, new_audiosegments
            )):
                rich.print(
                    f"[{i}] {audio_segment.starth} - "
                    f"{audio_segment.endh} ({audio_segment.duration})"
                )
                rich.print(f"[{i}] {text.text}")
        return ASRResult(texts=new_texts), new_audiosegments
