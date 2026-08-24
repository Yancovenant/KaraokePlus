import glob
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .config import config
from .text import safepath

__all__ = [
    "raise_for_permission",
    "search_for_path",
    "temp_filenames",
]

def raise_for_permission(path: str | Path, check: str = "all") -> None:
    """
    Checks file access and raises a PermissionError if the check fails.
    Modes available: 'read', 'write', 'execute', 'all'
    """
    path = Path(str(path))
    if path.suffix:
        path = path.parent
    path = str(path)
    permission_map = {
        "read": os.R_OK,
        "write": os.W_OK,
        "execute": os.X_OK,
        "all": os.R_OK | os.W_OK | os.X_OK  # Combines read, write, and execute
    }
    flag = permission_map.get(check.lower())
    if flag is None:
        raise ValueError("``check`` must be 'read', 'write', 'execute', or 'all'.")
    if not os.access(path, flag):
        raise PermissionError(f"Missing status: '{check}' permission denied for '{path}'.")

def search_for_path(filepath: str) -> str:
    """ Return the config directory if a path exists """
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
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)