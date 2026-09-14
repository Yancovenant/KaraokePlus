
import copy
from typing import Any, Literal, TypedDict


class KwargsMixin:
    def __init__(self, **kwargs):
        pass

    def update(self, **kwargs) -> dict:
        for key in self.__dict__:
            if key in kwargs:
                getattr(self, key).update(kwargs.pop(key))
        return kwargs


class ProcessorKwargs(TypedDict, total=False):
    return_tensors: Literal["pt", "np"]
    padding: bool


class ASRKwargs(TypedDict, total=False):
    _defaults: dict = {}

    processor_kwargs: ProcessorKwargs = {  # noqa: RUF012
        **ProcessorKwargs.__annotations__,
    }


def merge_kwargs(source: dict | ASRKwargs, kwargs: dict):
    user_kwargs = copy.deepcopy(kwargs)
    _defaults = getattr(source, "_defaults", source if isinstance(source, dict) else {})
    output_kwargs = copy.deepcopy(_defaults)
    leftovers = {}

    # Implicit Pass: Handle top level structures, and only update if not Explicit pass
    def _fill(data: dict, target_key:str, target_value: Any) -> bool:
        if isinstance(target_value, dict):
            return False
        applied = False
        for k, v in data.items():
            if k == target_key: # padding == padding
                data[k] = target_value
                applied = True
            elif isinstance(v, dict) and _fill(v, target_key, target_value):
                applied = True
        return applied
        
    for k in list(user_kwargs.keys()):
        if _fill(output_kwargs, k, user_kwargs[k]):
            user_kwargs.pop(k)
    
    # Explicit Pass: Handle Exact key structures
    # (e.g processor_kwargs={...})
    for k in list(user_kwargs.keys()):
        if k in output_kwargs:
            if isinstance(out_val:=output_kwargs[k], dict) and isinstance(user_val:=user_kwargs[k], dict):
                valid_val, unknown = merge_kwargs(out_val, user_kwargs.pop(k))
                output_kwargs[k] = valid_val
                if unknown: # empty dict wouldnt pass
                    leftovers[k] = unknown
            else:
                output_kwargs[k] = user_kwargs.pop(k)
    
    leftovers.update(user_kwargs)
    return output_kwargs, leftovers


if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    console.rule("Running Tests")

    # Test Purpose Only
    class Wav2Vec2Kwargs(ASRKwargs, total=False):
        _defaults = {  # noqa: RUF012
            "processor_kwargs": {
                "return_tensors": "pt",
                "padding": True,
                "text_kwargs": {
                    "padding": True,
                },
            },
            "common_kwargs": {},
        }
    
    console.print("1. Explicit Pass")
    out, left = merge_kwargs(Wav2Vec2Kwargs, {"padding": False})
    assert out["processor_kwargs"]["padding"] is False
    assert out["processor_kwargs"]["text_kwargs"]["padding"] is False
    assert left == {}, f"Failed Test 1 Leftovers: {left}"

    console.print("2. Implicit & Explicit Overrides")
    out, left = merge_kwargs(Wav2Vec2Kwargs, {
        "padding": False, 
        "processor_kwargs": {"padding": True}
    })
    assert out["processor_kwargs"]["padding"] is True                 # Explicit won
    assert out["processor_kwargs"]["text_kwargs"]["padding"] is True  # Cascade
    assert left == {}, f"Failed Test 2 Leftovers: {left}"

    console.print("3. Flat & Nested Leftovers")
    test_3_kwargs = {
        "hello": "iya",
        "padding": False, 
        "processor_kwargs": {
            "Hello": "yesy",
            "return_tensors": "np",
            "How about this": {
                "Nested 3": "Nested Value"
            },
            "text_kwargs": {
                "padding": False,
            }
        }
    }
    out, left = merge_kwargs(Wav2Vec2Kwargs, test_3_kwargs)
    assert out["processor_kwargs"]["return_tensors"] == "np"
    assert out["processor_kwargs"]["padding"] is False
    assert out["processor_kwargs"]["text_kwargs"]["padding"] is False
    expected_leftovers = {
        'processor_kwargs': {
            'Hello': 'yesy', 
            'How about this': {'Nested 3': 'Nested Value'}
        }, 
        'hello': 'iya'
    }
    assert left == expected_leftovers, f"Failed Test 3 Leftovers. Got: {left}"
    assert "text_kwargs" not in left.get("processor_kwargs", {}) # empty dict
