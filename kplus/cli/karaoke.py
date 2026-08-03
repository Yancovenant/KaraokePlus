import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kplus.environment import env
from kplus.pipelines import (
    AAD,
    AlignerAny,
    ReferenceAligner,
    Refiner,
    SeparatorMixin,
    Transcriber,
    get_track_file,
)
from kplus.pipelines.utils import AudioLoader
from kplus.tools.config import config
from kplus.tools.render import Render

from .command import Command

if TYPE_CHECKING:
    from kplus.pipelines.aad import AudioSegment
    from kplus.pipelines.transcriber import Result

logger = logging.getLogger(__name__)


class Karaoke(Command):
    """ Separate either input URL or file path into 2 different stems (vocals, instrumentals) """
    def run(self, args):
        self.parser.add_argument("-i", '--input', dest="filepath",
                                 help="The input file path or URL that needs to be make karaoke of, (mp4)")
        self.parser.add_argument("--lyricsfile", dest="lyricsfile",
                                 help="If input is not URL, and no lyrics path were given, default to multiplex only")
        group = self.parser.add_argument_group("Transcribe Options (Speech To Text)")
        group.add_argument("--use-cliptimestamp", action="store_true", help="Wheter to process audio segment by chunk or cliptimestamp")
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


        # Step 1 separate and maybe make it a wav first
        env.ffmpeg  # noqa: B018
        audio_file_path = f"{filepath.stem}.wav"
        # Hardcoded for now
        options = SimpleNamespace(modelname="demucs", overlap=0.75, segment=200, shifts=1)
        sep_class = SeparatorMixin.get_model(options)
        subprocess.run(["ffmpeg", "-y", "-i", str(filepath), "-vn", "-ar", str(sep_class.sr), "-ac", str(sep_class.ac), str(audio_file_path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sep_info = sep_class.separate(audio_file_path)
        del sep_class.model, sep_class
        env.clean()
        logger.info(f"Finished separating -- {audio_file_path}")
        logger.info(f">> Inst Path: {sep_info.inst_path}")
        logger.info(f">> Vocs Path: {sep_info.vocs_path}")
        logger.info(f">> SR: {sep_info.sr}")

        # Step 2 get audio segment
        aad_opts = SimpleNamespace(verbose=False, precision_ms=0.5, sr=sep_info.sr)
        _,audio_segments = AAD(aad_opts).get_audio_segments(sep_info.vocs_path)
        logger.info(f"Finished Getting Audio Segments -- {len(audio_segments)} segments")
        for aseg in audio_segments:
            logger.debug(f"Segment: {aseg.h_start:.3f} - {aseg.h_end:.3f} ({aseg.duration:.3f})")
        
        # At this point i think we wanna convert the sampling rate to be 16000 since both uses that?
        # 2.5 make the audio_np available and pass it then to the rest (using sr 16KHz for now for everything)
        audio_loader = AudioLoader(sep_info.vocs_path, samplerate=16000, channels=1)
        audio_np = audio_loader.audio_np

        # Step 3 Transcribe
        trans_class = Transcriber(verbose=True)
        trans_class.load_model("qwen")
        trans_result =  trans_class.transcribe(audio_np, audio_segments, info.lyrics)
        logger.info(f"Finished Transcribing -- {len(trans_result.segments)} segments")
        ref_segments, new_audio_segments = ReferenceAligner(verbose=True).get_reference_timestamp(
            trans_result, info.lyrics, audio_segments
        )

        ai_align_result = self._align_many(trans_class, audio_np, ref_segments, new_audio_segments)
        logger.info(f"Finished Alignment -- {len(ai_align_result)} AI Aligner, with {(len(seg) for ai_segs in ai_align_result for seg in ai_segs.segments)}")
        # Already cleaned
        refiner_class = Refiner(verbose=False, sr=16000, precision_ms=0.5) #0.5ms
        refine_result = refiner_class.refine_timestamp(audio_np, None, *ai_align_result, ref_segments=ref_segments, audio_segments=audio_segments)
        refine_result.to_lyrics_segment().populate_ass()
        logger.info(f"Finished Refinement and populating ass -- {len(refine_result.segments)} segments")

        # Last step rendering
        Render(with_ass=True).render(video_filepath=filepath, inst_path=sep_info.inst_path, duration=info.duration, result=refine_result)

    
    def _align_many(self, trans_class, audio_np, ref_segments, audio_segments):
        # qwen...
        model = trans_class.model
        fa_res_qwen = AlignerAny().align(model, audio_np, None, ref_segments, audio_segments)
        del model, trans_class.model, trans_class
        env.clean()
        # Whisper
        trans_class = Transcriber(verbose=True)
        model = trans_class.load_model("tiny", beamsize=10, max_threads=2)
        fa_res_whisper = AlignerAny().align(model, audio_np, None, ref_segments, audio_segments)
        del model, trans_class.model, trans_class
        env.clean()
        # MMS FA
        fa_res_def = AlignerAny().align(None, audio_np, None, ref_segments, audio_segments)
        env.clean()
        return (fa_res_qwen, fa_res_whisper, fa_res_def)
