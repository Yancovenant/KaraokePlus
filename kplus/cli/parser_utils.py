
from kplus.tools import RichArgumentParser

class Options:
    def add_options(cls, parser: RichArgumentParser) -> None:
        raise NotImplementedError()

class InputOptions(Options):
    @classmethod
    def add_options(cls, parser: RichArgumentParser, **kwargs) -> None:
        group = parser.add_argument_group("Input", "Specify the source media to process.")
        if kwargs.get("url_only"):
            group.add_argument("url", type=str, metavar="URL", help="The YouTube watch or share URL to download.")
        else:
            group.add_argument("filepath", type=str, metavar="PATH/URL", help="The input file path or URL that needs to be processed.")

class DownloadOptions(InputOptions):
    @classmethod
    def add_options(cls, parser: RichArgumentParser, **kwargs) -> None:
        super().add_options(parser, **kwargs)
        group = parser.add_argument_group("Download Options")
        group.add_argument("-c", "--cookiefile", default="cookies.txt", metavar="FILE",
                          help="Path to a Netscape formatted cookies.txt file. Required to bypass age-restrictions or access members-only content.")
