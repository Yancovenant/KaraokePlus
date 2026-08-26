
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

class SeparateOptions(DownloadOptions):
    @classmethod
    def add_options(cls, parser, **kwargs):
        super().add_options(parser, **kwargs)
        group = parser.add_argument_group("Separate Options")
        group.add_argument("--demucs", nargs="?", const="mdx_extra_q", default=None, metavar="MODEL",
                           help="Demucs model separator engine. (default: %(const)s)")
        group.add_argument("--uvr", nargs="?", const="UVR_MDXNET_KARA_2.onnx", default=None, metavar="MODEL",
                           help="Ultimate Vocal Remover / MDX model separator engine. (default: %(const)s)")
        group = parser.add_argument_group("Demucs Separation Options")
        group.add_argument("--segment", dest="segment", type=int, metavar="SECONDS", default=30,
                            help="Process audio in chunks of this length. Crucial for saving GPU VRAM on lower-end hardware. (default: %(default)s)")
        group.add_argument("--overlap", dest="overlap", type=float, metavar="RATIO", default=0.75,
                            help="The overlap percentage between audio splits (e.g., 0.25). Higher overlap reduces boundary artifacts but increases processing time. (default: %(default)s)")
        group.add_argument("--shifts", dest="shifts", type=int, metavar="N", default=1,
                            help="Number of random time-shifts. Improves SDR (Signal-to-Distortion) by up to 0.2 points, but multiplies processing time by N. (default: %(default)s)")    
        group.add_argument("--num-workers", type=int, metavar="N", default=0,
                           help="Number of jobs. This can increase memory usage but will be much faster when multiple cores are available. If not specified, will use the command line option. (default: %(default)s)")

class AudioDetectionOptions(SeparateOptions):
    @classmethod
    def add_options(cls, parser, **kwargs):
        super().add_options(parser, **kwargs)
        parser.add_argument("--separate", action="store_true",
                                 help="Wheter to run separation first, and use only its vocals")
        group = parser.add_argument_group("Audio Detection Options")
        group.add_argument("--precision-ms", type=int, metavar="MS", default=10,
                          help="Define how much 1 frame in millisecond. ")
        group.add_argument("--signal-overlap", type=float, metavar="RATIO", default=0.75,
                          help="Used to calculate frame length")
        group.add_argument("--use-filter", action="store_true", default=True,
                          help="Wheter to apply frequency filter (200, 5000) (Human Freq) to the audio being detect")
        

class TranscribeOptions(AudioDetectionOptions):
    @classmethod
    def add_options(cls, parser, **kwargs):
        super().add_options(parser, **kwargs)
        group = parser.add_argument_group("Transcribe Options")
        group.add_argument("--whisper", nargs="?", const="large-v3", default=None, metavar="MODEL",
                          help="Use Whisper model to transcribe")
        group.add_argument("--qwen", nargs="?", const="Qwen/Qwen3-ASR-1.7B", default=None, metavar="MODEL",
                          help="Use Qwen model to transcribe")
        group = parser.add_argument_group("Whisper Transcribe Options")
        group.add_argument("--max-threads", default=2)
        group.add_argument("--beam-size", default=10)
        group = parser.add_argument_group("Qwen Transcribe Options")
        group.add_argument("--max-inference-batch-size", default=-1)
        group.add_argument("--max-new-tokens", default=8192)
        group.add_argument("--num-beams", default=10)
        