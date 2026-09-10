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
from kplus.tools.audio import Audio
from kplus.tools.text import similarity

from .audio_aligner import AudioAligner
from .sequence_aligner import SequenceAligner
from .utils import Tokens

__all__ = [
    "LyricAligner"
]

class LyricAligner:
    """ Main For Lyrics Aligner """
    def __init__(self, *, raise_if_not_reliable: bool = True):
        self.sequence_aligner = SequenceAligner(raise_if_not_reliable=raise_if_not_reliable)
        self.audio_aligner = AudioAligner()
    
    def asr2ref(self,
        hypothesis: ASRResult,
        reference: str,
        audiosegments: list[AudioSegment],
        *,
        no_show: bool = True,
        **kwargs
    ) -> tuple[ASRResult, list[AudioSegment]]:
        ref_tokens, hyp_tokens = (
            self.sequence_aligner(
                Tokens.from_reference(reference),
                Tokens.from_asr(hypothesis.texts)
            )
        )
        ## 
        ##  TODO: Create multiple test here
        ##
        ref_by_audio = defaultdict(list)
        for i, audio_segment in enumerate(audiosegments):
            ref_by_audio[i] = []
            for j, token in enumerate(ref_tokens):
                if token.start is None: continue
                if overlap(audio_segment, token):
                    ref_by_audio[i].append(j)

        # Interpolation
        for audio_id, token_ids in ref_by_audio.items():
            if token_ids: ref_by_audio[audio_id] = list(range(min(token_ids), max(token_ids) + 1))

        # Extrapolation
        unused_gaps = {}
        for i, audio_segment in enumerate(audiosegments):
            unused_gaps[i] = (None, None)
            token_ids = ref_by_audio[i]
            if not token_ids: continue
            start_gap = ref_tokens[min(token_ids)].start - audio_segment.start
            end_gap = audio_segment.end - ref_tokens[max(token_ids)].end
            unused_gaps[i] = (start_gap, end_gap)
        last_word_idx = -1

        for i, audio_segment in enumerate(audiosegments):
            last_audio_id = len(audiosegments) - 1
            if i == last_audio_id and (unknown := list(range(last_word_idx + 1, len(ref_tokens)))):
                # This is if the last text is still left out
                prev_anchor_audio_id = max(k for k, v in ref_by_audio.items() if v)
                prev_start_gap, prev_end_gap = unused_gaps[prev_anchor_audio_id]
                if prev_end_gap > 0.4:
                    # Combining to previous anchor
                    ref_by_audio[prev_anchor_audio_id] = sorted(ref_by_audio[prev_anchor_audio_id] + unknown)
                else:
                    if prev_anchor_audio_id < i:
                        # Using exactly the next audio id
                        ref_by_audio[prev_anchor_audio_id+1] = sorted(unknown)
            token_ids = ref_by_audio[i]
            if not token_ids: continue

            start_gap, end_gap = unused_gaps[i]
            _, prev_end_gap = unused_gaps[i-1] if i > 0 else (None, None)
            next_start_gap, _ = unused_gaps[i+1] if i < last_audio_id else (None, None)
            if min(token_ids) > last_word_idx + 1 and (unknown := list(range(last_word_idx + 1, min(token_ids)))):
                unknown_text = " ".join(ref_tokens[j].word for j in unknown)
                known_text = " ".join(ref_tokens[j].word for j in token_ids)
                if prev_anchor_audio_id := next((j for j in range(i - 1, -1, -1) if ref_by_audio[j]), None):
                    _, anchor_end_gap = unused_gaps[prev_anchor_audio_id]
                    if anchor_end_gap > 1.0:
                        ref_by_audio[prev_anchor_audio_id] = sorted(ref_by_audio[prev_anchor_audio_id] + unknown)
                elif similarity(unknown_text, known_text) > 0.7:
                    # Similar in ratio.
                    ref_by_audio[i] = sorted(token_ids + unknown)
                else:
                    # Extrapolate to previous audio id
                    ref_by_audio[i] = sorted(token_ids + unknown + ref_by_audio[i-1])
                    ref_by_audio[i-1] = sorted(token_ids + unknown + ref_by_audio[i-1])
            token_ids = ref_by_audio[i]
            if i < last_audio_id and (next_token_ids := ref_by_audio[i+1]):
                if unknown := sorted(set(range(max(token_ids) + 1, min(next_token_ids)))):
                    if next_start_gap > end_gap:
                        # Unknown Belong to the next
                        ref_by_audio[i+1] = sorted(token_ids + unknown)
                    else:
                        # Unknown being merged next + current
                        ref_by_audio[i] = sorted(token_ids + unknown + next_token_ids)
                        ref_by_audio[i+1] = sorted(token_ids + unknown + next_token_ids)
                elif end_gap > 2.0 or next_start_gap > 1.0:
                    # No Unknown but the gap in either each segment is so long
                    # So we combine
                    ref_by_audio[i] = sorted(token_ids + unknown + next_token_ids)
                    ref_by_audio[i+1] = sorted(token_ids + unknown + next_token_ids)

            token_ids = ref_by_audio[i]
            last_word_idx = max(token_ids)

        def should_merge(i, j) -> bool:
            a_tokens = ref_by_audio[i]
            b_tokens = ref_by_audio[j]
            if not a_tokens or not b_tokens:
                return False
            return bool(not set(a_tokens).isdisjoint(set(b_tokens)))

        # Clustering
        clusters, current = [], None
        for i in range(len(audiosegments)):
            if not ref_by_audio[i]: continue
            if current is None:
                current = {"audio_ids": [i], "token_ids": list(ref_by_audio[i])}
                continue
            prev_i = current["audio_ids"][-1]
            if should_merge(prev_i, i):
                current["audio_ids"].append(i)
                current["token_ids"].extend(ref_by_audio[i])
                current["token_ids"] = sorted(set(current["token_ids"]))
            else:
                clusters.append(current)
                current = {"audio_ids": [i], "token_ids": list(ref_by_audio[i])}
        if current:
            clusters.append(current)
        new_audiosegments = []
        new_texts = []
        for i, cluster in enumerate(clusters):
            audio_ids = cluster["audio_ids"]
            token_ids = cluster["token_ids"]
            first = audiosegments[audio_ids[0]]
            last = audiosegments[audio_ids[-1]]
            new_audiosegments.append(AudioSegment(start=first.start, end=last.end,))
            words, langs = [], []
            for token_id in token_ids:
                token = ref_tokens[token_id]
                langs.append(token.language)
                words.append(WordTiming(start=token.start, end=token.end, score=token.score, word=token.word,))
            if not langs:
                langs = ["English"]
            new_texts.append(TextTiming(words=words, language=langs[0]))
        assert len(new_texts) == len(new_audiosegments)
        if env.verbose and not no_show:
            for i, (text, audio_segment) in enumerate(zip(new_texts, new_audiosegments)):
                rich.print(f"[{i}] {audio_segment.starth} - {audio_segment.endh} ({audio_segment.duration})")
                rich.print(f"[{i}] {text.text}")
        return ASRResult(texts=new_texts), new_audiosegments
        # # Not Used
        # datas = self.audio_aligner(ref_tokens, audiosegments)
        # new_audiosegments = []
        # results = []
        # for data in datas:
        #     min_audiosegment = audiosegments[min(data.audio_ids)]
        #     max_audiosegment = audiosegments[max(data.audio_ids)]
        #     new_audiosegments.append(AudioSegment(start=min_audiosegment.start, end=max_audiosegment.end))
        #     results.append(data.to_texttiming())
        # if env.verbose:
        #     for res, aseg in zip(results, new_audiosegments):
        #         rich.print(res.starth, res.endh, res.text)
        #         rich.print(aseg.starth, aseg.endh)
        #         rich.print("="*20)
        # return ASRResult(texts=results), new_audiosegments
    