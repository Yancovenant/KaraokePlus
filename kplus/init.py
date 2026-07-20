import kplus

from .release import Release
Release.print_banner()
kplus.Release = Release

from .environment import env
kplus.env = env

from .tools.config import config
kplus.config = config
