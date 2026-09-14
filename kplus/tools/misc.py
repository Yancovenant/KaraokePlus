from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any

__all__ = [
    "filter_known_kwargs",
]

def filter_known_kwargs(func: Callable, kwargs: dict[str, Any]) -> tuple[dict, dict]:
    """ Filter the given keyword arguments to only return the kwargs
        that binds to the function's signature and the unused one.
    """
    leftovers = set(kwargs)
    for p in signature(func).parameters.values():
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            leftovers.discard(p.name)
        elif p.kind == Parameter.VAR_KEYWORD:  # **kwargs
            leftovers.clear()
            break

    if not leftovers:
        return kwargs, {}
    used = {key: kwargs[key] for key in kwargs if key not in leftovers}
    leftovers = {key: kwargs[key] for key in leftovers}
    return used, leftovers


if __name__ == "__main__":
    print("Tools Misc Test")
