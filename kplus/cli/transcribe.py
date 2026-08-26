import logging
import sys
from pathlib import Path
from types import SimpleNamespace

#from kplus.pipelines import AAD, Transcriber, get_track_file
#from kplus.pipelines.aligner import ReferenceAligner
from kplus.tools import RichArgumentParser, config
from kplus.pipelines.asr import transcribe

from .command import Command
from .download import Download
from .parser_utils import TranscribeOptions

logger = logging.getLogger(__name__)

class Transcribe(Command):
    """ Whisper Transcribe given audio """

    @staticmethod
    def internal_transcribe_options(parser: RichArgumentParser):
        group = parser.add_argument_group("Transcribe Options")
        group.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="Initial Prompt for whisper")
        
    def _parse_config(self, args):
        TranscribeOptions.add_options(self.parser)
        opt = self.parser.parse_args(args)
        if not opt.filepath: self.parser.print_help(); sys.exit()
        config.parse_config(opt, setup_logging=True)
        return opt

    def _run_audio_detection(self, args):
        opt = self._parse_config(args)
        info = ensure_file(opt.filepath, no_lyrics=True)
        if opt.separate:
            separation_result = separate_song(info, **vars(opt))
            audiopath = separation_result.vocs_path
            sr = separation_result.sr
        else:
            audiopath = opt.filepath
            sr = Audio(str(opt.filepath)).samplerate()
        result = detect_audio_activity(audio=audiopath, sr=sr, **vars(opt))
        return info, audiopath, result
        
    def run(self, args):
        opt = self._parse_config(args)
        info, audiopath, audio_result = self._run_audio_detection(args)
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = "".join(f.readlines())
        transcibe(audiopath, audio_result, info.lyrics, **vars(opt))
        if info.lyrics is not None:
            reference_aligner = ReferenceAligner(opt.verbose)
            lyrics_segments, new_audio_segments = reference_aligner.get_reference_timestamp(
                transcriptions, info.lyrics, audio_segments)