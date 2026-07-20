import sys
import logging
import os

from pathlib import Path
from urllib.parse import urlparse

from .command import Command
from kplus.tools.config import config
from kplus.pipelines import AAD as _AAD
from kplus.environment import env

logger = logging.getLogger(__name__)

class AAD(Command):
    """ Audio Activity Detection using RMS, Pitch, Harmonics Accoustic
    """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path (.wav)")
        self.parser.add_argument("--visualize", dest="visualize", action="store_true")
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)

        _AAD(opt.visualize).detect(opt.filepath, sr=None)