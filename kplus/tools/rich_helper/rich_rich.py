from __future__ import annotations

# Needs Env
from kplus import env

# Setup
env.rich  # noqa: B018

class Rich:
    def __init__(self):
        self.console = self._console.Console()

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)

    def log(self, *args, **kwargs):
        """ Useful for logging output of rich renderables """
        with self.console.capture() as capture:
            self.console.print(*args, **kwargs)
        return self.Text.from_ansi(capture.get()).markup

    @property
    def inspect(self):
        from rich import inspect
        return inspect

    ##############
    # Module Level
    ##############
    @property
    def _console(self):
        from rich import console
        return console
    
    @property
    def _padding(self):
        from rich import padding
        return padding

    @property
    def _text(self):
        from rich import text
        return text

    @property
    def _panel(self):
        from rich import panel
        return panel

    @property
    def _table(self):
        from rich import table
        return table

    @property
    def _progress(self):
        from rich import progress
        return progress

    @property
    def _live(self):
        from rich import live
        return live
    

    #############
    # Class Level
    #############

    # Console
    @property
    def Group(self):
        return self._console.Group

    # Padding
    @property
    def Padding(self):
        return self._padding.Padding

    # Text
    @property
    def Text(self):
        return self._text.Text

    # Panel
    @property
    def Panel(self):
        return self._panel.Panel
    
    # Table
    @property
    def Table(self):
        return self._table.Table
        
    @property
    def Column(self):
        return self._table.Column

    # Renderables
    @property
    def Renderables(self):
        from rich.containers import Renderables  # type: ignore
        return Renderables

    # Progress
    @property
    def track(self):
        return self._progress.track

    # Live
    @property
    def Live(self):
        return self._live.Live
    
        
    # Explicit for logging `netsvc`
    @property
    def RichHandler(self):
        from rich.logging import RichHandler
        return RichHandler

    @property
    def RichRenderable(self):
        from rich.abc import RichRenderable
        return RichRenderable

    # Helper
    def make_progress(self, *, is_download: bool = True):
        _columns = [
            self._progress.DownloadColumn(),
            self._progress.TransferSpeedColumn(),
        ] if is_download else [
            self._progress.MofNCompleteColumn(),
        ]
        return self._progress.Progress(
            self._progress.SpinnerColumn("aesthetic"),
            self._progress.TextColumn("[progress.description]{task.description}"),
            self._progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            *_columns,
            self._progress.TimeRemainingColumn(),
            self._progress.TimeElapsedColumn(),
            expand=True,
        )
