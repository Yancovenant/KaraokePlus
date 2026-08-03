import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from kplus.pipelines import AAD, Transcriber, get_track_file
from kplus.pipelines.aligner import ReferenceAligner
from kplus.tools.config import config

from .command import Command

logger = logging.getLogger(__name__)

class Transcribe(Command):
    """ Whisper Transcribe given audio """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path or URL that needs to be make karaoke of, (mp4)")
        self.parser.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="Initial Prompt for whisper")
        group = self.parser.add_argument_group("Advanced Options")
        group.add_argument("--verbose", dest="verbose", action="store_true", help="Debug info more verbose")
        group.add_argument("--modelname", dest="modelname", default="large-v3", help="Which model used to transcribe, ``qwen`` or whisper model name")
        group.add_argument("--max-threads", default=2, type=int)
        group.add_argument("--beamsize", default=5, type=int)
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)
        info = get_track_file(opt.filepath, opt.lyricsfile is not None)
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = "".join(f.readlines())
        
        filepath = Path(info.filename)
        aad_opts = SimpleNamespace(verbose=False, precision_ms=0.5, sr=None)
        filtered_audio_np, audio_segments = AAD(aad_opts).get_audio_segments(filepath,)
        transcriber = Transcriber(opt.verbose)
        model_kwargs = {
            "beamsize": opt.beamsize,
            "max_threads": opt.max_threads,
        }
        transcriber.load_model(opt.modelname, **model_kwargs)
        transcriptions = transcriber.transcribe(filepath, audio_segments, info.lyrics)

        if info.lyrics is not None:
            reference_aligner = ReferenceAligner(opt.verbose)
            lyrics_segments, new_audio_segments = reference_aligner.get_reference_timestamp(
                transcriptions, info.lyrics, audio_segments)