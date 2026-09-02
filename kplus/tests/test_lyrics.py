import pytest
import rich
import pickle
from rich.console import Console
from pathlib import Path

import kplus
import kplus.init  # noqa: F401
from kplus.pipelines.lyrics import AudioAligner, LyricAligner, SequenceAligner
from kplus.pipelines.lyrics.utils import Tokens
from kplus.pipelines.utils import ASRResult, AudioSegment, TextTiming, WordTiming

console = Console()

path_data = Path(__file__).parent / Path("test_lyrics_data.pkl") 
console.print("loading data from path: ", path_data)
with open(str(path_data), "rb") as f:
    data = pickle.load(f)

@pytest.fixture
def reference():
    return data["lyrics"]

@pytest.fixture
def hypothesis() -> ASRResult:
    return ASRResult(texts=data["texts"])

@pytest.fixture
def audiosegments():
    return data["segments"]

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
    