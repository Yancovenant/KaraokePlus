
from .release import Release

__version__ = Release.version

from .environment import deprecated, env  # noqa: F401
from .tools.config import config  # noqa: F401