import argparse

from kplus.environment import env

class Rich:
    def __init__(self):
        env.rich; import rich
        from rich.console import Console
        self.console = Console()

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
    
    @property
    def _console(self):
        from rich import console
        return console

    @property
    def Group(self):
        return self._console.Group

    @property
    def _padding(self):
        from rich import padding
        return padding

    @property
    def Padding(self):
        return self._padding.Padding

    @property
    def _text(self):
        from rich import text
        return text

    @property
    def Text(self):
        return self._text.Text

    @property
    def _panel(self):
        from rich import panel
        return panel

    @property
    def Panel(self):
        return self._panel.Panel
    
    @property
    def _table(self):
        from rich import table
        return table
    
    @property
    def Table(self):
        return self._table.Table
        
    @property
    def Column(self):
        return self._table.Column

    @property
    def Renderables(self):
        from rich.containers import Renderables
        return Renderables
        
    # Explicit for logging
    @property
    def RichHandler(self):
        from rich.logging import RichHandler
        return RichHandler

rich = Rich()
import inspect as _ins
class RichArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        if file is None:
            file = rich.print
        _get_formatter_params = list(_ins.signature(self._get_formatter).parameters)
        if "file" in _get_formatter_params:
            formatter = self._get_formatter(file=file)
        else:
            formatter = self._get_formatter()
        try:
            help_text = self.format_help(formatter=formatter)
        except TypeError:
            # Backward compatibility for formatter classes that
            # do not accept the 'formatter' keyword argument.
            help_text = self.format_help()
        self._print_message(help_text, file)
    
class RichHelpFormatter(argparse.HelpFormatter):
    """ Implement Rich Module on argparse """
    
    def _fill_text(self, text, width, indent):
        print("Fill Text", text, width, indent)
        return ''.join(indent + line for line in text.splitlines(keepends=True))

    def _split_lines(self, text, width):
        print("Split Lines", text, width)
        return text.splitlines()

    def _get_help_string(self, action):
        print("Get Help String", action)
        return super()._get_help_string(action)

    def format_help(self):
        return ""