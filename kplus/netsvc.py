import logging
import logging.handlers
import os
import sys

from kplus.environment import env
from kplus.tools.config import config

logger = logging.getLogger(__name__)

class WatchedFileHandler(logging.handlers.WatchedFileHandler):
    def __init__(self, filename):
        self.errors = None  # py38
        super().__init__(filename)
        # Unfix bpo-26789, in case the fix is present
        self._builtin_open = None
    def _open(self):
        return open(self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors)

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, _NOTHING, DEFAULT = range(10)
#The background is set with 40 plus the number of the color, and the foreground with 30
#These are the sequences needed to get colored output
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLOR_PATTERN = f"{COLOR_SEQ}{COLOR_SEQ}%s{RESET_SEQ}"
LEVEL_COLOR_MAPPING = {
    logging.DEBUG: (BLUE, DEFAULT),
    logging.INFO: (GREEN, DEFAULT),
    logging.WARNING: (YELLOW, DEFAULT),
    logging.ERROR: (RED, DEFAULT),
    logging.CRITICAL: (WHITE, RED),}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        fg_color, bg_color = LEVEL_COLOR_MAPPING.get(record.levelno, (GREEN, DEFAULT))
        record.levelname = COLOR_PATTERN % (30 + fg_color, 40 + bg_color, record.levelname)
        return super().format(record)

class LogRecord(logging.LogRecord):
    def __init__(self, name, level, pathname, lineno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
        super().__init__(name, level, pathname, lineno, msg, args, exc_info, func=func, sinfo=sinfo, **kwargs)
        self.perf_info = "" # maybe add this later
        self.pid = os.getpid()
        

def setup_logger():
    if logging.getLogRecordFactory() is LogRecord:
        return
    logging.setLogRecordFactory(LogRecord)
    logging.captureWarnings(True)
    format = '%(asctime)s %(pid)s %(levelname)s %(name)s: %(message)s %(perf_info)s'
    handler = logging.StreamHandler()
    if (logf:=config["logfile"]):
        try:
            # We check we have the right location for the log files
            dirname = os.path.dirname(logf)
            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname)
            if os.name == 'posix':
                handler = WatchedFileHandler(logf)
            else:
                handler = logging.FileHandler(logf)
        except Exception:
            sys.stderr.write("ERROR: couldn't create the logfile directory. Logging to the standard output.\n")
    def is_a_tty(stream):
        return hasattr(stream, 'fileno') and os.isatty(stream.fileno())
    if os.name == 'posix' and isinstance(handler, logging.StreamHandler) and (is_a_tty(handler.stream) or os.environ.get("AOK_PY_COLORS")):
        formatter = ColoredFormatter(format)
    else:
        formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    pseudo_config = PSEUDOCONFIG_MAPPER.get(config['log_level'], [])
    logging_configurations = DEFAULT_LOG_CONFIGURATION + pseudo_config
    for logconfig_item in logging_configurations:
        loggername, level = logconfig_item.strip().split(':')
        level = getattr(logging, level, logging.INFO)
        _logger = logging.getLogger(loggername)
        _logger.setLevel(level)
    for logconfig_item in logging_configurations:
        logger.debug('logger level set: "%s"', logconfig_item)

DEFAULT_LOG_CONFIGURATION = [':INFO',]
PSEUDOCONFIG_MAPPER = {
    'debug': ['kplus:DEBUG'],
    'info': [],
    'warn': ['kplus:WARNING'],
    'error': ['kplus:ERROR'],
    'critical': ['kplus:CRITICAL'],}
