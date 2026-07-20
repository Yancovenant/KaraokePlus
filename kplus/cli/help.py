import textwrap
import kplus

from .command import PROG_NAME, Command, commands


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
        padding = max(len(cmd_name) for cmd_name in commands) + 2
        name_desc = [(cmd_name, (cmd.__doc__ or "").strip())
                     for cmd_name, cmd in sorted(commands.items())]
        command_list = "\n".join(f"    {name:<{padding}}{desc}" for name, desc in name_desc)
        print(Help.template.format(  # noqa: T201
            prog_name=PROG_NAME,
            version=kplus.Release.version,
            command_list=command_list,))