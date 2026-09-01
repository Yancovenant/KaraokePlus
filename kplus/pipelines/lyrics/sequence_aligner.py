import logging
from dataclasses import dataclass
from enum import StrEnum

from sequence_align.pairwise import needleman_wunsch_with_scores as nwws

from kplus import env
from kplus.tools import rich
from kplus.tools.text import similarity

from .utils import LyricAlignError, Tokens

logger = logging.getLogger(__name__)

__all__ = [
    "SequenceAligner",
]

class OP(StrEnum):
    """ Class Operator to hold operation type """
    I = "insert"
    S = "subtitute/replace"
    D = "delete"
    M = "match"


@dataclass(slots=True, frozen=True)
class SequenceResult:
    operations: list[dict]
    stats: dict[OP, int]
    ref_tokens: Tokens
    hyp_tokens: Tokens

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
        return self.substitutions + self.deletations + self.insertions

    @property
    def wer(self) -> float:
        denominator = self.matches + self.substitutions + self.deletations
        return self.errors / denominator if denominator else 0.0

    @property
    def reliable(self) -> bool:
        return self.wer <= 0.5

    def plot(self, chars_per_line: int = 80) -> None:
        """ Plot using rich module """
        stats_table = rich.Table.grid(rich.Column("Metric", style="b cyan"), rich.Column("Value", ratio=1, justify="right"), padding=(0, 2))
        stats_table.add_row("Matches (M)", rich.Text(str(self.matches), style="green"))
        stats_table.add_row("Substitutions (S)", rich.Text(str(self.substitutions), style="yellow"))
        stats_table.add_row("Deletions (D", rich.Text(str(self.deletations), style="red"))
        stats_table.add_row("Insertions (I)", rich.Text(str(self.insertions), style="magenta"))
        stats_table.add_row("─" * 17, "─" * 10)
        stats_table.add_row("Total Errors", rich.Text(str(self.errors), style="b red"))
        status_color = "green" if self.reliable else "red"
        stats_table.add_row("WER", rich.Text(f"{self.wer:.2%}", style=status_color))
        stats_table.add_row("Reliable", rich.Text(f"{'Yes' if self.reliable else 'No'}", style=status_color))
        rich.print(rich.Panel(stats_table, title=rich.Text("Sequence Stats", style="b color(4)"), border_style="color(4)"))
        rich.print()

        rich.console.rule("Alignment Breakdown", style="b blue")
        r_text = rich.Text("Ref : ", style="b cyan")
        h_text = rich.Text("Hyp : ", style="b cyan")
        o_text = rich.Text("OP  : ", style="b cyan")
        cur_len = 6
        for item in self.operations:
            r = (
                self.ref_tokens[item.get("ref_idx", None)].word
                if item.get("ref_idx", None) is not None else "-"
            )
            h = (
                self.hyp_tokens[item.get("hyp_idx", None)].word
                if item.get("hyp_idx", None) is not None else "-"
            )
            op_type = item.get("op", "-")

            width = max(len(r), len(h), len(op_type))
            r_padded = r.ljust(width)
            h_padded = h.ljust(width)
            o_padded = (
                op_type.name.ljust(width)
                if isinstance(op_type, OP) else
                op_type.ljust(width)
            )
            color = (
                "green" if op_type == OP.M else
                "yellow" if op_type == OP.S else 
                "red" if op_type == OP.D else 
                "magenta" if op_type == OP.I else
                "white"
            )
            if cur_len + width + 1 > chars_per_line and cur_len > 5:
                rich.print(r_text)
                rich.print(h_text)
                rich.print(o_text)
                rich.print()
                r_text = rich.Text("Ref : ", style="b cyan")
                h_text = rich.Text("Hyp : ", style="b cyan")
                o_text = rich.Text("OP  : ", style="b cyan")
                cur_len = 5
            r_text.append(r_padded + " ", style=color)
            h_text.append(h_padded + " ", style=color)
            o_text.append(o_padded + " ", style=color)
            cur_len += width + 1
        if cur_len > 5:
            rich.print(r_text)
            rich.print(h_text)
            rich.print(o_text)
            rich.print()

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
            map.append({
                "ref_idx": None if op == OP.I else ref_idx,
                "hyp_idx": None if op == OP.D else hyp_idx,
                "op": op
            })
            ref_idx += (op != OP.I)
            hyp_idx += (op != OP.D)
        result = SequenceResult(operations=map, stats=stats, ref_tokens=ref_tokens, hyp_tokens=hyp_tokens)
        if env.verbose:
            result.plot()
        if not result.reliable:
            logger.warning(f"Lyric Alignment may be inaccurate due to error rate of more than 50%: WER={result.wer * 100:.1f}%")
            raise LyricAlignError("Cannot continue as the lyric aligment may be inaccurate")
        return result
            
    def __call__(self, ref_tokens: Tokens, hyp_tokens: Tokens) -> tuple[Tokens, Tokens]:
        alignment = self.sequence_align(ref_tokens, hyp_tokens)
        for op in alignment.operations:
            if op["op"] in (OP.M, OP.S):
                ref, hyp = ref_tokens[op["ref_idx"]], hyp_tokens[op["hyp_idx"]]
                ref.start = float(round(hyp.start, 2))
                ref.end = float(round(hyp.end, 2))
                ref.score = float(round(hyp.score, 3))
                ref.language = str(hyp.language)
        return ref_tokens, hyp_tokens
    