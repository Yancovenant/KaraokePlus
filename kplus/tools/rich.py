from __future__ import annotations
import argparse

from kplus.environment import env

class Rich:
    def __init__(self):
        env.rich; import rich  # noqa: B018, F401, I001
        from rich.console import Console  # type: ignore
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
        from rich.containers import Renderables  # type: ignore
        return Renderables

    @property
    def inspect(self):
        from rich import inspect
        return inspect
        
    # Explicit for logging
    @property
    def RichHandler(self):
        from rich.logging import RichHandler
        return RichHandler

    # Progress
    @property
    def _progress(self):
        from rich import progress
        return progress

    def make_progress(self):
        return self._progress.Progress(
            self._progress.SpinnerColumn(),
            self._progress.TextColumn("[progress.description]{task.description}"),
            self._progress.BarColumn(),
            self._progress.TextColumn("{task.percentage:>5.1f}%"),
            self._progress.DownloadColumn(),
            self._progress.TransferSpeedColumn(),
            self._progress.TimeRemainingColumn(),
        )

rich = Rich()
import sys as _sys
from gettext import gettext as _, ngettext
import re as _re

class RichArgumentParser(argparse.ArgumentParser):
    def __init__(self, prog = None, usage = None, description = None, epilog = None,
                     parents = [], formatter_class = argparse.HelpFormatter, prefix_chars = "-",
                     fromfile_prefix_chars = None, argument_default = None, conflict_handler = "error",
                     add_help = True, allow_abbrev = True, exit_on_error = True):
            # Check Version of argparse
            import inspect as _inspect
            try:
                self._has_file_param = "file" in list(_inspect.signature(super()._get_formatter).parameters)
            except (AttributeError, ValueError):
                self._has_file_param = False
            super().__init__(prog, usage, description, epilog, parents, formatter_class, prefix_chars, fromfile_prefix_chars, argument_default, conflict_handler, add_help, allow_abbrev, exit_on_error)

    def format_help(self, formatter=None):
        if formatter is None: formatter = self._get_formatter()
        # description
        formatter.add_text(self.description)
        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)
        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
        # epilog
        formatter.add_text(self.epilog)
        # determine help from format above
        help = formatter.format_help()
        return rich.Panel(rich.Group(*help.renderables), padding=1, title=f"{self.prog} (Help)", border_style="color(10)")
    
    def _get_formatter(self, file=None):
            if self._has_file_param: return super()._get_formatter(file)
            return super()._get_formatter()

    def print_usage(self, file = None):
        if file is None:
            file = rich.console
        return super().print_usage(file)
    
    def print_help(self, file=None):
        if file is None:
            file = rich.console
        return super().print_help(file)
    
    def _print_message(self, message, file = None):
        if message:
            file = file or _sys.stderr
            if isinstance(file, rich._console.Console):
                file.print(message)
            else:
                try:
                    file.write(message)
                except (AttributeError, OSError):
                    pass
    
    def error(self, message):
        self.print_usage(rich.console)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, _('%(prog)s: error: %(message)s\n') % args)

class RichHelpFormatter(argparse.HelpFormatter):
    _root_section: _Section
    _current_section: _Section

    class _Section(argparse.HelpFormatter._Section):
        def __init__(self, formatter, parent, heading = None):
            super().__init__(formatter, parent, heading)
            self.formatter: RichHelpFormatter
            
        def format_help(self):
            if self.heading is not argparse.SUPPRESS and self.heading is not None:
                heading = _('%(heading)s:') % dict(heading=self.heading)
            else: heading = ''
            heading = rich.Text(heading, style="b color(11)")
            parts = [heading]
            item_parts = []
            action_table = rich.Table.grid(
                rich.Column(width=self.formatter._action_max_length, style="b dim", no_wrap=True),
                rich.Column(ratio=1, style="color(15)", justify="all"), expand=True, padding=(0, 1)
            )
            for func, args in self.items:
                res = func(*args)
                if isinstance(res, dict) and self.parent is not None:
                    action_table.add_row(res["action_header"], res["help_text"])
                else:
                    item_parts.append(res)
            if action_table.row_count > 0:
                action_table.add_row("", "")
                parts.append(action_table)
            if item_parts:
                parts.extend(item_parts)
            return rich.Group(*parts)

    def _format_text(self, text):
        if '%(prog)' in text:
            text = text % dict(prog=self._prog)
        text = rich.Text(text, style="b color(15)")
        text_space = rich.Text("")
        return rich.Group(text, text_space)
    
    def _format_usage(self, usage, actions, groups, prefix):
        usage_row, action_usage = [], None
        usage_table = rich.Table.grid(rich.Column(width=self._action_max_length, style="b color(15)"), rich.Column(ratio=1, style="color(14)", justify="left"), expand=True, padding=(0, 1))
        if prefix is None: prefix = _('Usage: ')
        usage_row.append(prefix)
        if usage is not None: usage = usage % dict(prog=self._prog)
        elif usage is None and not actions: usage = '%(prog)s' % dict(prog=self._prog)
        elif usage is None:
            usage = '%(prog)s' % dict(prog=self._prog)
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings: optionals.append(action)
                else: positionals.append(action)
            try:
                # manage different version of argparse
                action_usage = self._format_actions_usage(optionals + positionals, groups)
            except AttributeError:
                parts, pos_start = self._get_actions_usage_parts(actions, groups)
                action_usage = " ".join([*parts])
        if usage is not None and "kplus-bin" in usage:
            usage = "kplus-bin" + f" [color(11)]{usage.strip("kplus-bin ")}[/]"
        usage_row.append(usage)
        if action_usage is not None:
            usage_table.add_column(ratio=1, style="dim")
            usage_row.append(action_usage)
        usage_table.add_row(*usage_row)
        return usage_table

    def _format_action(self, action):
        action_part_dict = {}
        action_part_dict["action_header"] = self._format_action_invocation(action)
        action_part_dict["help_text"] = self._expand_help(action)
        action_part_dict["subaction"] = []
        for subaction in self._iter_indented_subactions(action):
            action_part_dict["subaction"].append(self._format_action(subaction))
        if action_part_dict["subaction"]:
            rich.print("Format Action ->")
            rich.print(action_part_dict)
            rich.print(self._current_section, self._current_section.parent, self._current_section.heading)
            rich.print("*"*50)
        return action_part_dict

    def format_help(self):
        help = self._root_section.format_help()
        if isinstance(help, rich.Group): return help
        else: return rich.Group(*help)
