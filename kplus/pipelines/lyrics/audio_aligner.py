
from kplus import env
from kplus.pipelines.utils import AudioSegment
from kplus.tools import rich

from .utils import AudioAlignment, Token, Tokens

__all__ = [
    "AudioAligner",
]

class AudioAligner:
    """ Align by audio """
    def _prepare_alignment(self, tokens: Tokens, audiosegments: list[AudioSegment]):
        datas = [
            AudioAlignment(line_idx=line_idx, tokens=group_tokens, audio_ids=[])
            for line_idx, group_tokens in tokens.groups.items()
        ]
        for audio_id, segment in enumerate(audiosegments):
            for data in datas:
                if any(token.start is not None and token.end is not None
                    and not (segment.end < token.start or segment.start > token.end
                ) for token in data.tokens):
                    data.audio_ids.append(audio_id)
        return datas

    def _interpolate_lines(self, datas: list[AudioAlignment], audiosegments: list[AudioSegment]) -> list[AudioAlignment]:
        datas.sort(key=lambda x: x.line_idx)
        for i, data in enumerate(datas):
            if len(data.audio_ids) > 0: continue
            prev_idx = next((j for j in range(i - 1, -1, -1) if datas[j].audio_ids), None,)
            next_idx = next((j for j in range(i + 1, len(datas)) if datas[j].audio_ids), None,)
            prev_idx = prev_idx or 0
            next_idx = next_idx or len(datas) - 1
            dropped = datas[prev_idx+1:next_idx]
            if env.verbose:
                rich.print(f">> {i} Dropped: {prev_idx} - {next_idx}")
                for drop in dropped:
                    rich.print(f">>>> {drop.audio_ids}")
            assert len(datas[next_idx].audio_ids) > 0
            assert len(datas[prev_idx].audio_ids) > 0
            min_audio_id = max(datas[prev_idx].audio_ids)
            max_audio_id = min(datas[next_idx].audio_ids)
            if env.verbose:
                rich.print(f">> {i} Audio Interpolate {min_audio_id} - {max_audio_id}")
            assert min_audio_id < max_audio_id
            for drop in dropped:
                assert len(drop.audio_ids) == 0
                drop.audio_ids = list(range(min_audio_id, max_audio_id+1))
                if env.verbose:
                    rich.print(f">>>> {drop.audio_ids}")
        # assert
        for i, data in enumerate(datas):
            assert len(data.audio_ids) > 0
            if env.verbose:
                rich.print(f"{i}: {data.audio_ids}")
        if env.verbose:
            rich.print("="*20)
        return datas

    def _interpolate_words(self, datas: list[AudioAlignment], audiosegments: list[AudioSegment]) -> list[AudioAlignment]:
        def is_populated(token: Token) -> bool:
            return token.start is not None and token.end is not None
        for i, data in enumerate(datas):
            if not is_populated(data.tokens[0]):
                prev_audio_id = (
                    max(datas[i-1].audio_ids) if i > 0 else
                    max(0, min(data.audio_ids) - 1)
                )
                min_audio_id = min(min(data.audio_ids), prev_audio_id)
                max_audio_id = min(data.audio_ids)
                data.tokens[0].start = audiosegments[min_audio_id].start
                data.tokens[0].end = audiosegments[max_audio_id].end
                if min_audio_id not in data.audio_ids:
                    data.audio_ids.append(min_audio_id)
                if max_audio_id not in data.audio_ids:
                    data.audio_ids.append(max_audio_id)
            if not is_populated(data.tokens[len(data.tokens) - 1]):
                next_audio_id = (
                    min(datas[i+1].audio_ids) if i < len(datas) - 1 else
                    min(len(audiosegments) - 1, max(data.audio_ids) + 1)
                )
                max_audio_id = max(max(data.audio_ids), next_audio_id)
                min_audio_id = max(data.audio_ids)
                data.tokens[len(data.tokens) - 1].start = audiosegments[min_audio_id].start
                data.tokens[len(data.tokens) - 1].end = audiosegments[max_audio_id].end
                if min_audio_id not in data.audio_ids:
                    data.audio_ids.append(min_audio_id)
                if max_audio_id not in data.audio_ids:
                    data.audio_ids.append(max_audio_id)
        assert data.tokens[0].start is not None, data
        assert data.tokens[len(data.tokens) - 1] is not None
        for i, data in enumerate(datas):
            min_start = min(t.start for t in data.tokens if t.start is not None)
            max_end = max(t.end for t in data.tokens if t.end is not None)
            for j, token in enumerate(data.tokens):
                if env.verbose:
                    rich.print(f"{i} - {j}: {token}")
                if token.start is not None and token.end is not None:
                    assert token.start <= token.end
                    continue
                # just interpolate it fully
                token.start = min_start
                token.end = max_end
        return datas
        
    def __call__(self, ref_tokens: Tokens, audiosegments: list[AudioSegment]) -> list[AudioAlignment]:
        audiosegments.sort(key=lambda x: x.start)
        datas: list[AudioAlignment] = self._prepare_alignment(ref_tokens, audiosegments)
        datas: list[AudioAlignment] = self._interpolate_lines(datas, audiosegments)
        datas: list[AudioAlignment] = self._interpolate_words(datas, audiosegments)
        return datas
        