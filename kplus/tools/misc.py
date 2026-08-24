from collections.abc import Callable
from inspect import Parameter, signature
from pathlib import Path

__all__ = [
    "filter_known_kwargs",
    "is_file",
]

def filter_known_kwargs(func: Callable, kwargs: dict[str, ...]) -> tuple[dict, dict]:
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

def is_file(path: str | Path | None) -> bool | Path:
    if path is None:
        return False
    path = Path(str(path))
    path = path.expanduser()
    if path.is_file():
        return path
    else:
        return False

