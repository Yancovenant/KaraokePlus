import sys
from pathlib import Path

from rich import inspect

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import kplus.init
from kplus.pipelines.utils import *


def test_word_timing():
    word = WordTiming(start=2, end=2, word="OK")
    inspect(word)
    word.start = 3
    word.end = 5
    inspect(word)
    print(word)
    


if __name__ == "__main__":
    test_word_timing()





