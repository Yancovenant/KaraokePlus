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
    print(round(word.start))
    old_pickled_state = {
        'word': 'hello',
        'start': 1.23,
        'end': 4.56,
        'score': 0.99
    }
    print("--- Starting Test ---")
    test_obj = WordTiming.__new__(WordTiming)
    test_obj.__setstate__(old_pickled_state)
    print("\n--- Verifying Data ---")
    print(f"Expected start: 1.23, Got: {test_obj.start}")
    print(f"Expected end: 4.56, Got: {test_obj.end}")
    print("\n--- Testing Setters ---")
    test_obj.start = 2.00


if __name__ == "__main__":
    test_word_timing()





