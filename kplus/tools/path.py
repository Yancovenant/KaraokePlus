import os

from pathlib import Path

__all__ = [
    "raise_for_permission",
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