import logging
import sys

from kplus.pipelines import (
    align2ref,
    separate_song,
    detect_audio_activity,
    ensure_file,
    transcribe,
    align,
)
from kplus import config
from kplus.tools.render import Render

from .command import Command
from .parser_utils import KaraokeOptions

logger = logging.getLogger(__name__)


class Karaoke(Command):
    """ Create a karaoke subtitle ready video. """

    def _parse_config(self, args):
        KaraokeOptions.add_options(self.parser)
        opt = self.parser.parse_args(args)
        if not opt.filepath:
            self.parser.print_help()
            sys.exit()
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
        audio_result = detect_audio_activity(
            audio=audiopath, sr=sr, **vars(opt)
        )
        return info, audiopath, audio_result, separation_result

    def run(self, args):
        opt = self._parse_config(args)
        info, audiopath, audio_result, separation_result = self._run_audio_detection(opt)
        if not info.lyrics and not opt.lyricsfile:
            logger.warning("Running karaoke without lyrical subtitle")
        if opt.lyricsfile is not None:
            with open(opt.lyricsfile, "rt", encoding="utf-8") as f:
                info.lyrics = f.readlines()

        result, with_ass = None, False
        if info.lyrics:
            asr_result = transcribe(
                audiopath,
                audio_result.segments,
                info.lyrics, **vars(opt)
            )
            ref_result, new_audiosegments = align2ref(
                asr_result,
                info.lyrics,
                audio_result.segments
            )

            result = align(
                audiopath,
                ref_result,
                new_audiosegments,
                **vars(opt)
            )
            result.to_line_idx(info.lyrics)
            with_ass = True
        (
            Render(with_ass=with_ass)
            .render(
                video_filepath=info.filepath,
                inst_path=separation_result.inst_path,
                duration=info.duration,
                result=result.populate_ass()
            )
        )
