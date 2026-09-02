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

path_data = Path(__file__).parent / Path("test_align_data.pkl") 
console.print("loading data from path: ", path_data)
with open(str(path_data), "rb") as f:
    data = pickle.load(f)

@pytest.fixture
def reference():
    return data["lyrics"]

@pytest.fixture
def transcriptions():
    return data["ref_result"]

@pytest.fixture
def audiosegments():
    return data["segments"]


def test_qwen_prepare_data(transcriptions, audiosegments):
    console.print(f"{len(transcriptions.texts)} - {len(audiosegments)}")
    for hyp, seg in zip(transcriptions.texts, audiosegments):
        if not (seg.end < hyp.start or seg.start > hyp.end):
            safe_start = max(0, max(min(hyp.start, seg.start), hyp.start - 1.0) - 0.5)
            safe_end = min(300, min(max(hyp.end, seg.end), hyp.end + 1.0) + 0.5)
            console.print("OK", safe_start, safe_end)
        else:
            console.print("Wtf", hyp)