from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime

from kplus import env
from kplus.tools import config, rich

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    Text = rich.Text
    Console = rich._console.Console
    ConsoleRenderable = rich._console.ConsoleRenderable
    from rich._log_render import FormatTimeCallable
    from rich.text import TextType
    Group = rich.Group
    from rich.traceback import Traceback

class WatchedFileHandler(logging.handlers.WatchedFileHandler):
    def __init__(self, filename):
        self.errors = None  # py38
        super().__init__(filename)
        # Unfix bpo-26789, in case the fix is present
        self._builtin_open = None
        
    def _open(self):
        return open(self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors)


class RichLoggingHandler(rich.RichHandler):
    """ Custom rendered format for rich handler. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_time = kwargs.get("show_time", True)
        self.show_level = kwargs.get("show_level", True)
        self.show_path = kwargs.get("show_path", True)
        self.time_format = kwargs.get("log_time_format", "[%x %X]")
        self.omit_repeated_times = kwargs.get("omit_repeated_times", True)
        self.level_width = kwargs.get("level_width", None)
        self._last_time: Text | None = None
        if self.enable_link_path and (self.console.is_terminal and (env.is_colab or env.is_kaggle)):
            # Check wheter environment is supported it
            logger.warning("Setting up clickable path link isn't supported on a jupyter cloud server")
            self.enable_link_path = False
        
    def _log_rich_render(self,
        console: Console,
        renderables: Iterable[ConsoleRenderable],
        log_time: datetime | None = None,
        time_format: str | FormatTimeCallable | None = None,
        level: TextType = "",
        path: str | None = None,
        line_no: int | None = None,
        link_path: str | None = None,
        pid: int = 0,
        name: str | None = None,
    ) -> Group:
        title = rich.Text(overflow="ellipsis", no_wrap=True)
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format): log_time_display = time_format(log_time)
            else: log_time_display = log_time.strftime(time_format)
            if log_time_display == self._last_time and self.omit_repeated_times:
                time_text = " " * len(log_time_display)
            else:
                time_text = log_time_display
                self._last_time = log_time_display
            title.append(rich.Text(time_text + " ", style="log.time"))
        title.append(rich.Text(f"[PID: {pid}] ", style="log.path"))
        if self.show_path:
            title.append(rich.Text("─ ", style="log.path"))
            path_text = rich.Text(style="log.path")
            if name: path_text.append(name, style=f"link file://{link_path}" if link_path else "")
            elif path: path_text.append(path, style=f"link file://{link_path}" if link_path else "")
            if line_no:
                path_text.append(f":{line_no}", style=f"link file://{link_path}#{line_no}" if link_path else "")
            pad_length = max(0, self.console.width - len(title) - len(path_text))
            title.append(rich.Text.assemble(" " * pad_length, path_text))
        body = rich.Table.grid(rich.Column(style="log.level", width=self.level_width), rich.Column(ratio=1, style="log.message"), expand=True, padding=(0, 1))
        level_text = rich.Text(style="log.level")
        if self.show_level:
            level_text.append(" ↳ ", style="log.level")
            if isinstance(level, rich.Text): level_text.append(level)
            else: level_text.append(rich.Text(level, style="log.level"))
        else: level_text.append(" ↳ Message:")
        body.add_row(level_text, rich.Renderables(renderables))
        return rich.Group(title, body)
        
    def render(
        self,
        *,
        record: LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        # TODO: Should this be monkey patch instead on the module level
        # of rich._log_render
        path = os.path.basename(record.pathname)
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)
        log_renderable = self._log_rich_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
            pid=record.pid,
            name=record.name,
        )
        return log_renderable

    def render_message(self, record, message, **kwargs):
        if isinstance(message, rich.RichRenderable):
            return message
        return super().render_message(record, message, **kwargs)

class RichLoggingFormatter(logging.Formatter):
    def format(self, record, **kwargs):
        if isinstance(record.msg, rich.RichRenderable):
            return record.msg
        return super().format(record, **kwargs)
        
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
    file_format = '%(asctime)s %(pid)s %(levelname)s %(name)s: %(message)s %(perf_info)s'
    rich_format = '%(message)s %(perf_info)s'
    #handler = logging.StreamHandler()
    handler = None
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
            handler.setFormatter(logging.Formatter(file_format))
        except Exception:
            sys.stderr.write("ERROR: couldn't create the logfile directory. Logging to the standard output.\n")
    if handler is None:
        handler = RichLoggingHandler(
            rich_tracebacks=True, markup=True,
            show_path=True, show_time=True, show_level=True,
            omit_repeated_times=False, console=rich.console,
        )
        handler.setFormatter(RichLoggingFormatter(rich_format))
    if (root_logger:=logging.getLogger()).hasHandlers() and (env.is_colab or env.is_kaggle):
        root_logger.handlers.clear()
    root_logger.addHandler(handler)
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
