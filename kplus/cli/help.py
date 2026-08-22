import textwrap
import kplus

from .command import PROG_NAME, Command, commands, load_internal_commands
from kplus.tools.rich import rich

class Help(Command):
    """ Display the list of available commands """
    template = textwrap.dedent("""\
        usage: {prog_name} <command> [...]

        All in One Karaoke {version}
        Available commands:

        {command_list}

        Use '{prog_name} separate --help' for regular separate options.
        Use '{prog_name} <command> --help' for other individual commands options.
    """)
    def run(self, args):
        load_internal_commands()
        padding = max(len(cmd_name) for cmd_name in commands) + 2
        header = rich.Table.grid(rich.Column(style="b color(15)"), rich.Column(ratio=1), expand=True, padding=(0, 1))
        header.add_row("Version", f": [b color(14)]Karaoke+ {kplus.Release.version}[/]")
        header.add_row("", "")
        header.add_row("Usage", f": [color(14)]{PROG_NAME}[/] [b color(11)]<command>[/] [dim][...][/]")
        header.add_row("", "")
        header.add_row("Available commands", ":")
        com_list_text = []
        cmd_body = rich.Table.grid(rich.Column(style="b color(11)"), rich.Column(ratio=1, style="dim"), expand=True, padding=(0, 1))
        for cmd_name, cmd in sorted(commands.items()):
            cmd_body.add_row(f" • {cmd_name.ljust(padding)}:", f"{(cmd.__doc__ or '').strip()}")
        footer = rich.Table.grid(rich.Column(ratio=1), expand=True)
        footer.add_row("")
        footer.add_row(f"[dim]💡 Use [color(14)]'{PROG_NAME} separate --help'[/] for regular separate options.[/]")
        footer.add_row(f"[dim]💡 Use [color(14)]'{PROG_NAME} <command> --help'[/] for other individual commands options.[/]")
        rich.print(
            rich.Panel(
                rich.Group(header,
                    cmd_body, footer
                ),
                title="Help",
                padding=1, border_style="color(10)"
            )
        )