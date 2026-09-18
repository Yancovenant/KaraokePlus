from __future__ import annotations

import argparse
import sys as _sys
from gettext import gettext as _

from . import rich


class RichHelpFormatter(argparse.HelpFormatter):
    _root_section: _Section
    _current_section: _Section

    class _Section(argparse.HelpFormatter._Section):
        def __init__(self, formatter, parent, heading = None):
            super().__init__(formatter, parent, heading)
            self.formatter: RichHelpFormatter
            
        def format_help(self):
            if self.heading is not argparse.SUPPRESS and self.heading is not None:
                heading = _('%(heading)s:') % {'heading': self.heading}
            else:
                heading = ''
            
            heading = rich.Text(heading, style="b color(11)")
            parts = [heading]
            item_parts = []

            action_table = rich.Table.grid(
                rich.Column(style="b dim", ratio=1),
                rich.Column(style="color(15)", ratio=2), expand=True, padding=(0, 1)
            )

            for func, args in self.items:
                if self.parent is not None and func == self.formatter._format_text:
                    res = func(*args, style="dim")
                    spacer = rich.Text(" ")
                    res = rich.Group(res, spacer)
                else:
                    res = func(*args)

                if isinstance(res, dict) and self.parent is not None:
                    action_header = rich.Text.from_ansi(res["action_header"],)
                    action_header.pad_left(1)
                    action_table.add_row(action_header, rich.Text(res["help_text"]))
                    for subaction in res["subaction"]:
                        subaction_header = rich.Text.from_ansi(subaction["action_header"])
                        subaction_header.pad_left(2)
                        action_table.add_row(subaction_header, rich.Text(subaction["help_text"]))
                else:
                    item_parts.append(res)
                
            if item_parts:
                parts.extend(item_parts)
            
            if action_table.row_count > 0:
                action_table.add_row("", "")
                parts.append(action_table)
            
            if action_table.row_count == 0 and not item_parts:
                return ""
        
            if self.parent is not None:
                heading_title = parts.pop(0)
                return rich.Panel(
                    rich.Group(*parts),
                    title=heading_title,
                    title_align="left",
                    border_style="color(4)",
                    padding=1
                )
            else:
                return rich.Group(*parts)

    def _format_text(self, text, style="b color(15)"):
        if '%(prog)' in text:
            text = text % {'prog': self._prog}
        
        text = rich.Text(text, style=style)
        return text
    
    def _format_usage(self, usage, actions, groups, prefix):
        usage_row, action_usage = [], None
        usage_table = rich.Table.grid(
            rich.Column(style="b color(15)", ratio=1),
            expand=True, padding=(0, 1)
        )
        if prefix is None:
            prefix = _('Usage: ')
        
        text_prefix = rich.Text.from_ansi(prefix, style="b color(15)")

        if usage is not None:
            usage = usage % {'prog': self._prog}
        elif usage is None and not actions:
            usage = '%(prog)s' % {'prog': self._prog}  # noqa: UP031
        elif usage is None:
            usage = '%(prog)s' % {'prog': self._prog}  # noqa: UP031
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
        
        if usage is not None:
            if "kplus-bin" in usage:
                usage_text = rich.Text("kplus-bin", style="color(14)")
                usage_text.append(usage.strip("kplus-bin"), style="color(11)")
            else:
                usage_text = rich.Text(usage, style="color(14)")
        
        text_prefix.append(usage_text)
        usage_row.append(text_prefix)
        if action_usage is not None:
            usage_table.add_column(ratio=1, style="dim")
            usage_row.append(rich.Text.from_ansi(action_usage))
        
        usage_table.add_row(*usage_row)
        return usage_table

    def _format_action(self, action):
        action_part_dict = {}
        action_part_dict["action_header"] = self._format_action_invocation(action)
        action_part_dict["help_text"] = self._expand_help(action) if (action.help and action.help.strip()) else ""
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
        if isinstance(help, rich.Group):
            return help
        elif isinstance(help, list):
            return rich.Group(*help)
        else:
            return rich.Group(help)


class RichArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        prog = None,
        usage = None,
        description = None,
        epilog = None,
        parents = [],
        formatter_class = RichHelpFormatter,
        prefix_chars = "-",
        fromfile_prefix_chars = None,
        argument_default = None,
        conflict_handler = "error",
        add_help = True,
        allow_abbrev = True,
        exit_on_error = True,
        *args,
        **kwargs
    ):
        # Check Version of argparse
        import inspect as _inspect
        try:
            self._has_file_param = "file" in list(_inspect.signature(super()._get_formatter).parameters)
        except (AttributeError, ValueError):
            self._has_file_param = False
        super().__init__(prog, usage, description, epilog, parents, formatter_class, prefix_chars, fromfile_prefix_chars, argument_default, conflict_handler, add_help, allow_abbrev, exit_on_error, *args, **kwargs)

    def format_help(self, formatter=None):
        if formatter is None:
            formatter = self._get_formatter()
        
        # description
        formatter.add_text(self.description)

        # usage
        formatter.add_usage(
            self.usage,
            self._actions,
            self._mutually_exclusive_groups
        )
        
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

        if isinstance(help, rich.Group):
            renderables = rich.Group(*help.renderables)
        else:
            renderables = help
        
        return rich.Panel(renderables, padding=1, title=f"{self.prog} (Help)", border_style="color(10)")
    
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
