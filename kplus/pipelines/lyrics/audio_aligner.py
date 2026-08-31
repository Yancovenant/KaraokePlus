
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
            prev_line, next_line = None, None
            prev_gap, next_gap = 0, 0
            for j in range(i - 1, -1, -1):
                if (prev_line:=datas[j]).audio_ids: break
                prev_gap += 1
            for j in range(i + 1, len(datas)):
                if (next_line:=datas[j]).audio_ids: break
                next_gap += 1
            start_idx = i - prev_gap
            end_idx = i + next_gap
            dropped = [datas[k] for k in range(start_idx, end_idx + 1)]
            min_audio_id = max(prev_line.audio_ids) if prev_line is not None else 0
            max_audio_id = min(next_line.audio_ids) if next_line is not None else len(audiosegments) - 1
            assert min_audio_id <= max_audio_id, "This is safe to assume that there something wrong with the alignment"
            for drop in dropped: drop.audio_ids = list(range(min_audio_id, max_audio_id + 1))
        # assert
        for data in datas:
            assert len(data.audio_ids) > 0
        return datas

    def _interpolate_words(self, datas: list[AudioAlignment], audiosegments: list[AudioSegment]) -> list[AudioAlignment]:
        def is_populated(token: Token) -> bool:
            return token.start is not None and token.end is not None
        for i, data in enumerate(datas):
            if not is_populated(data.tokens[0]):
                if i > 0: min_audio_id = max(datas[i-1].audio_ids)
                else: min_audio_id = max(0, min(data.audio_ids) - 1)
                max_audio_id = min(data.audio_ids)
                min_audiosegment = audiosegments[min_audio_id]
                max_audiosegment = audiosegments[max_audio_id]
                data.tokens[0].start = min_audiosegment.start
                data.tokens[0].end = max_audiosegment.end
                if min_audio_id not in data.audio_ids:
                    data.audio_ids.append(min_audio_id)
                if max_audio_id not in data.audio_ids:
                    data.audio_ids.append(max_audio_id)
            if not is_populated(data.tokens[len(data.tokens) - 1]):
                if i < len(data.tokens) - 1: max_audio_id = min(datas[i+1].audio_ids)
                else: max_audio_id = min(len(audiosegments) - 1, max(data.audio_ids) + 1)
                min_audio_id = max(data.audio_ids)
                min_audiosegment = audiosegments[min_audio_id]
                max_audiosegment = audiosegments[max_audio_id]
                data.tokens[len(data.tokens) - 1].start = min_audiosegment.start
                data.tokens[len(data.tokens) - 1].end = max_audiosegment.end
                if min_audio_id not in data.audio_ids:
                    data.audio_ids.append(min_audio_id)
                if max_audio_id not in data.audio_ids:
                    data.audio_ids.append(max_audio_id)
        assert data.tokens[0].start is not None, data
        assert data.tokens[len(data.tokens) - 1] is not None
        for data in datas:
            min_start = min(t.start for t in data.tokens if t.start is not None)
            max_end = max(t.end for t in data.tokens if t.end is not None)
            for token in data.tokens:
                if token.start is not None and token.end is not None: continue
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
        