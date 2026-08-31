import logging
import sys

from kplus.pipelines import (
    align2ref,
    detect_audio_activity,
    ensure_file,
    separate_song,
    transcribe,
)

#from kplus.pipelines import AAD, Transcriber, get_track_file
#from kplus.pipelines.aligner import ReferenceAligner
from kplus.tools import RichArgumentParser, config

from .command import Command
from .parser_utils import TranscribeOptions

logger = logging.getLogger(__name__)

class Transcribe(Command):
    """ Whisper Transcribe given audio """
    def _parse_config(self, args):
        TranscribeOptions.add_options(self.parser)
        self.parser.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="Initial Prompt for whisper")
        opt = self.parser.parse_args(args)
        if not opt.filepath: self.parser.print_help(); sys.exit()
        config.parse_config(opt, setup_logging=True)
        return opt

    def _run_audio_detection(self, opt):
        info = ensure_file(opt.filepath, no_lyrics=True)
        if opt.separate:
            separation_result = separate_song(info, **vars(opt))
            audiopath = separation_result.vocs_path
            sr = separation_result.sr
        else:
            from kplus.tools.audio import Audio
            audiopath = opt.filepath
            sr = Audio(str(opt.filepath)).samplerate()
        result = detect_audio_activity(audio=audiopath, sr=sr, **vars(opt))
        return info, audiopath, result
        
    def run(self, args):
        opt = self._parse_config(args)
        info, audiopath, audio_result = self._run_audio_detection(opt)
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = "".join(f.readlines())
        result = transcribe(audiopath, audio_result.segments, info.lyrics, **vars(opt))
        if info.lyrics is not None:
            ref_result, new_audiosegments = align2ref(result, info.lyrics, audio_result.segments)