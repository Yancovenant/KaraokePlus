import kplus

from .release import Release
kplus.Release = Release

from .environment import env, deprecated
kplus.env = env
kplus.deprecated = deprecated

from .tools.config import config
kplus.config = config

kplus.__version__ = Release.version