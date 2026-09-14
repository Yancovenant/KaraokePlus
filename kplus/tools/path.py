import glob
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "resolve_path",
    "safepath",
    "search_for_path",
    "temp_filenames",
]


def search_for_path(filepath: str) -> str:
    """ Return the config directory if a path exists """
    from .config import config
    # Resolve to ( data_dir / * artist / requestedpath.stem )
    search_pattern = str(Path(config["data_dir"]).expanduser() / "*" / safepath(filepath))
    matching_files = glob.glob(search_pattern)
    if matching_files:
        return Path(matching_files[0]).parent
    return None


@contextmanager
def temp_filenames(count: int, delete=True):
    """ Yield temporary file names based on the requested counts
    """
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)  # noqa: SIM115
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def resolve_path(path: str | Path | None) -> bool | Path:
    """ Checks if a path points to an existing file and returns its resolved Path object. """
    if path is None:
        return False
    path = Path(str(path))
    path = path.expanduser()
    if path.is_file():
        return path
    else:
        return False


def safepath(s: str) -> str:
    return "".join([c for c in s if c.isalpha() or c.isdigit() or c in ' _-']).strip()

if __name__ == "__main__":
    print("-- Path Test --")
    print("1. Search For Path")
    print("2. resolve_path")
    print("3. safepath")
    print(f">> Result {safepath("KaraokePlus123_@#(!@*)_!@$")}")