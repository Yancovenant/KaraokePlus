
from dataclasses import dataclass

from .utils import LyricAlignError, Tokens

class OP(StrEnum):
    """ Class Operator to hold operation type """
    I = "insert"
    S = "subtitute/replace"
    D = "delete"
    M = "match"


@dataclass(slots=True, frozen=True)
class SequenceResult:
    operations: list[dict]
    stats: list

    @property
    def substitutions(self) -> int:
        return self.stats[OP.S]

    @property
    def deletations(self) -> int:
        return self.stats[OP.D]

    @property
    def insertions(self) -> int:
        return self.stats[OP.I]

    @property
    def matches(self) -> int:
        return self.stats[OP.M]

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        denominator = self.matches + self.substitutions + self.deletions
        return self.errors / denominator if denominator else 0.0

    @property
    def reliable(self) -> bool:
        return self.wer <= 0.5

    def plot(self) -> None:
        """ Plot using rich module """


class SequenceAligner:
    """ Align by Sequence """
    @staticmethod
    def sequence_align(ref_tokens: Tokens, hyp_tokens: Tokens) -> SequenceResult:
        def mwws_score(a:str, b: str) -> float:
            if a == b: return 2.0 # Match exactly
            if similarity(a, b) > 0.6: return 1.0
            return -3.0
        refs, hyps = nwws(ref_tokens.cleans, hyp_tokens.cleans, gap="-", score_fn=mwws_score, indel_score=-1)
        stats = {OP.M: 0, OP.S: 0, OP.D: 0, OP.I: 0}
        map, ref_idx, hyp_idx = [], 0, 0
        for ref, hyp in zip(refs, hyps):
            op = (OP.I if ref == "-" else (OP.D if hyp == "-" else (OP.M if ref == hyp else OP.S)))
            stats[op] += 1
            map.append(dict(
                ref_idx=None if op == OP.I else ref_idx,
                hyp_idx=None if op == OP.D else hyp_idx,
                op=op
            ))
            ref_idx += (op != OP.I)
            hyp_idx += (op != OP.D)
        result = SequenceResult(operations=map, stats=stats)
        if not result.reliable:
            logger.warning(f"Lyric Alignment may be inaccurate due to error rate of more than 50%: WER={result.wer * 100:.1f}%")
            raise LyricAlignError("Cannot continue as the lyric aligment may be inaccurate")
        if env.verbose:
            result.plot()
        return result
            
    def __call__(self, ref_tokens: Tokens, hyp_tokens: Tokens) -> Tuple[Tokens, Tokens]:
        alignment = self.sequence_align(ref_tokens, hyp_tokens)
        for op in aligment.operations:
            if op["op"] in (OP.M, OP.S):
                ref, hyp = ref_tokens[op["ref_idx"]], hyp_tokens[op["hyp_idx"]]
                ref.start = float(round(hyp.start, 2))
                ref.end = float(round(hyp.end, 2))
                ref.score = float(round(hyp.score, 3))
                ref.language = str(hyp.language)
        return ref_tokens, hyp_tokens
    