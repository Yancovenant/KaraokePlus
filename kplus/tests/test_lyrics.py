import pytest
import rich
from rich.console import Console

import kplus
import kplus.init  # noqa: F401
from kplus.pipelines.lyrics import AudioAligner, LyricAligner, SequenceAligner
from kplus.pipelines.lyrics.utils import Tokens
from kplus.pipelines.utils import ASRResult, AudioSegment, TextTiming, WordTiming

console = Console()


@pytest.fixture
def reference():
    return """
hello world
this is a test
of lyric alignment
Lorem ipsum dolor sit amet,
in non qui sint esse nisi.
Eu in cupidatat aliqua veniam aliquip
nisi aliqua labore commodo deserunt reprehenderit
"""

@pytest.fixture
def hypothesis() -> ASRResult:
    return ASRResult(texts=[
        TextTiming(words=[
            WordTiming(word="hello", start=11.05, end=12.5, score=0.0), # Low Score
            WordTiming(word="world", start=12.5, end=13.95, score=0.5),
        ], language="en"),
        TextTiming(words=[
            WordTiming(word="thank", start=14.5, end=14.6, score=0.1),
            WordTiming(word="you!", start=14.6, end=15.00, score=0.1),
            WordTiming(word="this", start=15.35, end=18.25, score=0.05), # Low Score
            WordTiming(word="is", start=18.25, end=19.20, score=0.1),
            WordTiming(word="a", start=19.20, end=21.5, score=0.1),
            WordTiming(word="test", start=21.5, end=22.25, score=0.01), # Low Score
        ], language="en"),
        TextTiming(words=[
            WordTiming(word="of", start=22.30, end=24.25, score=0.02), # Low Score
            WordTiming(word="lyric", start=25.5, end=26.00, score=0.1),
            WordTiming(word="alignment", start=26.00, end=27.00, score=0.3),
        ], language="en"),
        TextTiming(words=[
            WordTiming(word="lorem", start=29.10, end=32.25, score=0.02), # Low Score
            WordTiming(word="ipsum", start=32.5, end=34.00, score=0.1),
            WordTiming(word="dolor", start=34.00, end=35.00, score=0.3),
            WordTiming(word="sit", start=34.00, end=35.00, score=0.3),
            WordTiming(word="amet", start=34.00, end=35.00, score=0.3),
        ], language="en"),
        TextTiming(words=[
            WordTiming(word="in", start=35.10, end=38.25, score=0.02), # Low Score
            WordTiming(word="non", start=38.5, end=40.00, score=0.02), # Low Score
            WordTiming(word="qui", start=41.00, end=41.50, score=0.02), # Low Score
            WordTiming(word="sint", start=41.50, end=42.00, score=0.02), # Low Score
            WordTiming(word="esse", start=42.00, end=43.00, score=0.02), # Low Score
            WordTiming(word="nisi", start=43.00, end=45.00, score=0.02), # Low Score
        ], language="en"),
        TextTiming(words=[
            WordTiming(word="Eu", start=45.10, end=46.50, score=0.02), # Low Score
            WordTiming(word="in", start=46.50, end=47.00, score=0.1),
            WordTiming(word="cupidatat", start=47.00, end=47.50, score=0.3),
            WordTiming(word="aliqua", start=47.50, end=48.00, score=0.3),
            WordTiming(word="veniam", start=48.00, end=49.00, score=0.03), # Low Score
        ], language="en"),
        TextTiming(words=[
            WordTiming(word="aliquip", start=50.00, end=51.00, score=0.3),
            WordTiming(word="nisi", start=52.00, end=53.00, score=0.3),
            WordTiming(word="aliqua", start=54.00, end=55.00, score=0.3),
            WordTiming(word="labore", start=56.00, end=57.00, score=0.3),
            WordTiming(word="commodo", start=58.00, end=59.00, score=0.3),
            WordTiming(word="deserunt", start=60.00, end=62.00, score=0.3),
            WordTiming(word="reprehenderit", start=62.00, end=65.00, score=0.03),  # Low Score
        ], language="en")
    ])

@pytest.fixture
def audiosegments():
    return [
        AudioSegment(start=10.00, end=11.00), # 0: Empty / Adlib
        AudioSegment(start=11.20, end=14.00), # 1: Real
        AudioSegment(start=14.00, end=15.00), # 2: Empty
        AudioSegment(start=15.00, end=18.00), # 3: Real
        AudioSegment(start=18.00, end=22.00), # 4: Real
        AudioSegment(start=22.00, end=27.00), # 5: Empty
        AudioSegment(start=28.00, end=30.00), # 6: Empty
        AudioSegment(start=30.00, end=35.00), # 7: Real
        AudioSegment(start=35.00, end=38.00), # 8: Real
        AudioSegment(start=39.00, end=42.00), # 9: Real
        AudioSegment(start=42.00, end=45.00), # 10: Real
        AudioSegment(start=45.00, end=48.00), # 11: Real
        AudioSegment(start=48.00, end=51.00), # 12: Real
        AudioSegment(start=51.00, end=55.00), # 13: Real
        AudioSegment(start=55.00, end=62.00), # 14: Real
        AudioSegment(start=62.00, end=67.00), # 15: Real
    ]

@pytest.fixture
def reference_tokens(reference: str) -> Tokens:
    return Tokens.from_reference(reference)

@pytest.fixture
def hypothesis_tokens(hypothesis: ASRResult) -> Tokens:
    return Tokens.from_asr(hypothesis.texts)

def test_exact_sequence_alignment(reference_tokens, hypothesis_tokens):
    aligner = SequenceAligner()
    result = aligner.sequence_align(reference_tokens, hypothesis_tokens)
    assert result.reliable

def test_sequence_alignment_populates_reference_timestamps(reference_tokens, hypothesis_tokens,):
    aligner = SequenceAligner()
    ref_tokens, hyp_tokens = aligner(reference_tokens, hypothesis_tokens)
    for token in ref_tokens:
        if token.score is None:
            assert token.start == None and token.end == None
        else:
            assert token.score >= 0.1

def test_audio_alignment(reference_tokens, hypothesis_tokens, audiosegments,):
    sequence_aligner = SequenceAligner()
    audio_aligner = AudioAligner()

    ref_tokens, _ = sequence_aligner(reference_tokens, hypothesis_tokens,)

    result = audio_aligner(ref_tokens, audiosegments,)
    for data in result:
        assert data.tokens[0].start is not None
        assert data.tokens[len(data.tokens) - 1].end is not None
        console.print(data.tokens)
        console.print(data.audio_ids)
        console.print("="*20)

def test_full_alignment(reference, hypothesis, audiosegments):
    result, new_audiosegments = LyricAligner().asr2ref(hypothesis, reference, audiosegments)
    for text in result.texts:
        assert text.start is not None
        assert text.end is not None
    assert len(result.texts) == len(new_audiosegments)
    