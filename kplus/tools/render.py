import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from kplus import config, env
from kplus.pipelines.utils import ASRResult

logger = logging.getLogger(__name__)

class Render:
    def __init__(self, with_ass: bool = True):
        self.with_ass = with_ass
        env.fonts  # noqa: B018
        
    def render(
        self,
        video_filepath: str | Path,
        inst_path: str | Path,
        duration: float,
        result: ASRResult,
        output_path: str | None = None
    ) -> Path:
        env.rich, env.ffmpeg # noqa: B018
        import rich # type: ignore # noqa: I001
        console = rich.console.Console()
        try:
            safe_title = Path(str(video_filepath)).stem
            safe_title = "".join(c for c in safe_title if c.isalnum() or c in ' -_').rstrip()
            if output_path is None: output_path = f"(Karaoke)_{safe_title}.mkv"
            render_cmds = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_filepath), "-i", str(inst_path),]
            if self.with_ass:
                ass_content = ASRResult.ASS_HEADER + "\n".join(text.ass_event for text in result.texts)
                ass_path = Path(config.work_dir) / f"ass_{safe_title}.ass"
                ass_path.write_text(ass_content, encoding="utf-8-sig")
                scale_filter = "fps=30,scale=if(gt(iw/ih\\,16/9)\\,-1\\,1280):if(gt(iw/ih\\,16/9)\\,720\\,-1):flags=fast_bilinear,crop=1280:720"
                filters_list = [
                    "-filter_complex", f"[0:v]{scale_filter}[base];[base]ass=filename={ass_path.as_posix()}[subbed];[subbed]copy[outv]",
                    "-map", "[outv]",
                ]
                video_encoding = ["-c:v", "h264_nvenc", "-cq", "34", "-preset", "p1", "-tune", "hq", "-rc", "vbr", "-b:v", "0", "-maxrate", "4M", "-bufsize", "8M",]
            else:
                filters_list = ["-map", "0:v:0",]
                video_encoding = ["-c:v", "copy",]
            render_cmds += [
                *filters_list,
                "-map", "1:a:0", "-map", "0:a:0",
                "-metadata", f"title={safe_title} (Karaoke+ by iantirta Version)", "-metadata", "artist=kplus",
                "-metadata:s:a:0", "title=Karaoke", "-disposition:a:0", "default",
                "-metadata:s:a:1", "title=Original Song", "-disposition:a:1", "none",
                "-t", str(duration),
                *video_encoding,
                "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path),
            ]
            logger.info(f">> Rendering video... {safe_title}")
            logger.debug(" ".join(c for c in render_cmds))
            res = subprocess.run(render_cmds, check=True, capture_output=True)
            if res.returncode == 0:
                logger.info(f">> Successfully rendered: {output_path}")
            else:
                console.print(f"[red]⚠ FFmpeg failed to render video: {res.stderr.decode()}[/]")
        except Exception as e:
            console.print(f"[yellow]⚠ Video Can't be rendered as to why: {e}[/]")
            raise
        return output_path
