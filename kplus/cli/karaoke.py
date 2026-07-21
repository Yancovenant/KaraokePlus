import sys
import logging

from pathlib import Path
from typing import TYPE_CHECKING

from .command import Command
from kplus.tools.config import config
from kplus.pipelines import SeparationDemucs, get_track_file, \
    Aligner, AAD, Transcriber
from kplus.environment import env
from kplus.tools.render import Render

if TYPE_CHECKING:
    from kplus.pipelines.transcriber import Result
    from kplus.pipelines.aad import AudioSegment

logger = logging.getLogger(__name__)


class Karaoke(Command):
    """ Separate either input URL or file path into 2 different stems (vocals, instrumentals) """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path or URL that needs to be make karaoke of, (mp4)")
        self.parser.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="If input is not URL, and no lyrics path were given, default to multiplex only")
        self.parser.add_argument("--max-threads", dest="max_threads", type=int, help="max thread for running whisper")
        opt, unknown = self.parser.parse_known_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
        config.parse_config(unknown, setup_logging=True)
        info = get_track_file(opt.filepath, opt.lyricsfile is not None)
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = f.readlines()
        filepath = Path(info.filename)
        separation_model = SeparationDemucs(overlap_ratio=0.75,
                                            segment_size=200, shifts=1)
        separation_info = separation_model.separate(filepath, "all")
        logger.info(f"Finished separating {filepath}")
        del separation_model.model, separation_model
        env.clean()
        audio_segments  : AudioSegment = AAD(False).get_audio_segments(separation_info.vocal_tensor,
                sr=separation_info.sr)
        # At this point i think we wanna convert the sampling rate to be 16000 since both uses that?
        transcribe_model = Transcriber(max_threads=opt.max_threads)
        transcriptions  : Result = transcribe_model.transcribe(separation_info.vocal_tensor, sr=separation_info.sr,
                                                                audio_segments=audio_segments, lyrics=info.lyrics)
        del transcribe_model.model, transcribe_model
        env.clean()
        aligner_model = Aligner()
        aligner_info = aligner_model.main(separation_info.vocal_tensor,
                separation_info.sr, info.lyrics, audio_segments, transcriptions)
        del aligner_model.model, aligner_model
        env.clean()
        karaoke_data = []
        for seg in aligner_info.segments:
                word_list = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
                karaoke_data.append({"text": seg.text, "words": word_list})
        Render().render(None, info.title, info.filename, separation_info.inst_path, info.duration, karaoke_data)