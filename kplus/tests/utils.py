import sys
from pathlib import Path

from rich import inspect

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import kplus.init
from kplus.pipelines.utils import *


def test_load_old_pickle_wordtiming():
    old_picke_dict = {'word': 'hello','start': 1.23,'end': 4.56,'score': None}
    word = WordTiming.__new__(WordTiming)
    assert word.start is None and word.score is None and word.end is None
    word.__setstate__(old_picke_dict)
    assert word.start == 1.23 and word.end == 4.56

def test_crud_wordtiming():
    word = WordTiming(start=2, end=3, word="OK")
    word.start = 3
    assert word._start == word.start

def test_word_timing():
    test_load_old_pickle_wordtiming()
    test_crud_wordtiming()


if __name__ == "__main__":
    test_word_timing()





