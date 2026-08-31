
from enum import StrEnum

from kplus.tools.text import normalizekaldi, RomajiPhonetic, similarity

from .sequence_aligner import SequenceAligner
from .utils import Tokens

@dataclass(slots=True)
class Token:
    word: str
    start: float | None
    end: float | None
    language: str | None
    line_idx: int | None
    score: float | None

    _clean: str | ... = ...
    _phone: RomajiPhonetic | ... = ...

    @property
    def clean(self) -> str:
        if self._clean is not ...: return self._clean
        self._clean = normalizekaldi(self.word)
        return self._clean

    @property
    def phone(self) -> RomajiPhonetic:
        if self._phone is not ...: return self._phone
        self._phone = get_phonetic(self.word)
        return self._phone

@dataclass(slots=True)
class Tokens:
    tokens: list
    lines: list | None

    _cleans: list[str] | ... = ...
    _groups: dict[int, list[Token]] | ... = ...
    
    @classmethod
    def from_reference(cls, reference: str) -> Tokens:
        lines = [line.strip() for line in reference.split("\n") if line.strip() and not line.startswith('[')]
        tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            Token(
                word=word, score=None,
                start=None, end=None, 
                line_idx=i, language=None,
            ) for i, line in enumerate(lines)
            for word in line.split()
        ]
        return cls(tokens=tokens, lines=lines)

    @classmethod
    def from_asr(cls, texts: list[TextTiming]) -> Tokens:
        tokens = [ # List of WordTiming [(Word),(Word),(Word)]
            Token(
                word=w.word, score=w.score,
                start=w.start, end=w.end,
                idx=None, language=t.language,
            ) for t in texts for w in t.words if w.score > 0.1
        ]
        return cls(tokens=tokens, lines=None)

    @property
    def cleans(self):
        if self._cleans is not ...: return self._cleans
        self._cleans = [t.clean for t in self.tokens]
        return self._cleans

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    @property
    def groups(self):
        if self._groups is not ...: return self._groups
        _groups = defaultdict(Token)
        for token in self:
            _groups[token.line_idx].append(token)
        self._groups = _groups
        return self._groups



    
def populatenum(a, b, *attr) -> None:
    for att in attr:
        try:
            setattr(a, att) = safefloat(getattr(b, att))
        except Exception as err:
            logger.error("Couldnt able to populate %s: on %r | %r", att, a, b)
            raise

def populateby(currents, sources, **attr) -> None:
    for curr, src in zip(currents, sources):
        populatenum(curr, src, **attr)

def safefloat(num: float, round: 2) -> float:
    """ While were at it, round it also, since SSA timing in 10ms """
    return float(round(num, round))

class AudioAligner:
    def fill_audio_ids(ref_tokens: Tokens, audiosegments: list[AudioSegment]):
        for i, (start, end) in enumerate((s.start, s.end) for s in audiosegments):
            for j, tokens in ref_tokens.groups.items():
                for token in tokens:
                    if not (end < token.start or start > token.end):
                        if i not in token.audio_ids:
                            token.audio_ids.append(i)
                
                
    def interpolate_texts():
        for i, tokens in ref_tokens.groups.items():
            pass
        
    def __call__(self, ref_tokens: Tokens, audiosegments: list[AudioSegment]) -> ...:
        starts, ends = ((s.start, s.end) for s in audiosegments)
        fill_audio_ids(ref_tokens)
        interpolate_texts()
        

@dataclass
class AudioResult:
    pass

@dataclass
class AudioAlignment:
    line_idx: int
    tokens: Tokens # per word
    audio_ids: list = []

class AudioAligner:
    """ Align by audio """
    def _prepare_alignment(self, tokens, starts, ends):
        assert len(starts) == len(ends), f"Starts and ends differ {starts} | {ends}"
        datas = []
        for line_idx, group_tokens in tokens.groups.items():
            datas.append(AudioAlignment(line_idx=line_idx, tokens=group_tokens))
        for i, (s, e) in enumerate(zip(starts, ends)):
            for data in datas:
                for token in data.tokens:
                    if not (e < token.start or s > token.end):
                        data.audio_ids.append(i)
        return datas

    def _interpolate_lines(self, datas: list[AudioAlignment], audiosegments):
        datas.sort(key=lambda x: x.line_idx)
        for i, data in enumerate(datas):
            if len(data.audio_ids) > 0: continue
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
            assert min_audio_id <= max_audio_id f"This is safe to assume that there something wrong with the alignment"
            for drop in dropped: drop.audio_ids = list(range(min_audio_id, max_audio_id + 1))
        # assert
        for data in datas:
            assert len(data.audio_ids) > 0
        return datas

    def _interpolate_words(self, datas: list[AudioAlignment], audiosegments):
        def is_none(token: Token) -> bool:
            return token.start is not None and token.end is not None
        for i, data in enumerate(datas):
            if is_none(data.tokens[0]):
                if i > 0: audio_id = max(datas[i-1].audio_ids)
                else: audio_id = max(0, min(data.audio_ids) - 1)
                audiosegment = audiosegments[audio_id]
                data.tokens[0].start = audiosegment.start
                data.tokens[0].end = audiosegment.end
            if is_none(data.tokens[len(data.tokens) - 1]):
                if i < len(data.tokens) - 1: audio_id = min(datas[i+1].audio_ids)
                else: audio_id = min(len(data.tokens) - 1, max(data.audio_ids) + 1)
                audiosegment = audiosegments[audio_id]
                data.tokens[len(data.tokens) - 1].start = audiosegment.start
                data.tokens[len(data.tokens) - 1].end = audiosegment.end
            assert data.tokens[0].start is not None
            assert data.tokens[len(data.tokens) - 1] is not None
            for token in enumerate(data.tokens):
                if token.start is not None and token.end is not None: continue
                min_start = min(t.start for t in data.tokens)
                max_end = max(t.end for t in data.tokens)
                # just interpolate it fully
                token.start = min_start
                token.end = max_end
        return datas
        
    def __call__(self, ref_tokens: Tokens, audiosegments: list[AudioSegment]) -> ...:
        audiosegments.sort(key=lambda x: x.start)
        starts, ends = ((s.start, s.end) for s in audiosegments)
        datas: list[AudioAlignment] = self._prepare_alignment(ref_tokens, starts, ends)
        datas = self._interpolate_lines(datas, audiosegments)
        datas = self._interpolate_words(datas, audiosegments)
        return datas
        

class LyricAligner:
    """ Main For Lyrics Aligner """
    def __init__(self):
        self.sequence_aligner = SequenceAligner
        self.audio_aligner = AudioAligner
    
    def asr2ref(self,
        hypothesis: ASRResult,
        reference: str,
        audiosegments: list[AudioSegment]
    ) -> ASRResult:
        ref_tokens, hyp_tokens = (
            self.sequence_aligner(
                Tokens.from_reference(reference),
                Tokens.from_asr(hypothesis.texts)
            )
        )
        audio_alignment = self.audio_aligner(ref_tokens, audiosegments)
        new_audio_segment = []
        result = []
        for data in datas:
            min_audiosegment = audiosegments[min(data.audio_ids)]
            max_audiosegment = audiosegment[max(data.audio_ids)]
            new_audio_segment.append(AudioSegment(start=min_audiosegment.start, end=max_audiosegment.end))
            result.append(TextTiming(
                words=[
                    WordTiming(start=t.start, end=t.end, score=t.score) t for t in data.tokens
                ]
            ))
        return ASRResult(texts=result)