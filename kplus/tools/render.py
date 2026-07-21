import re
import logging
import subprocess
import sys

from typing import Optional
from pathlib import Path

from kplus.tools.config import config
from kplus.environment import env

logger = logging.getLogger(__name__)

class Render:
    ASS_STYLE: list = [
        "Style: Lat_Duet,Montserrat Bold,120,&H0000A5FF&,&H00FFFFFF&,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1",
        "Style: CJK_Duet,Noto Sans CJK SC,120,&H0000A5FF&,&H00FFFFFF&,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1",
    ]
    ASS_HEADER: str = (
        "[Script Info]\n"
        "Title: Karaoke+ by iantirta\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{chr(10).join(ASS_STYLE)}"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )
    def __init__(self, with_ass: bool = True):
        env._ensure_fonts_installed, env.ffmpeg
        self.RE_CJK = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+')
        self.with_ass = with_ass
        self.scale_filter = "fps=30,scale=if(gt(iw/ih\\,16/9)\\,-1\\,1280):if(gt(iw/ih\\,16/9)\\,720\\,-1):flags=fast_bilinear,crop=1280:720"

    def _convert_ssa_time(self, t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02}:{s:05.2f}"

    def build_ass(self, karaoke_data):
        events, prev_end = [], 0.0
        for line in karaoke_data:
            words = line.get("words", [])
            if not words: continue

            b_start = words[0]["start"]
            display_start = max(0.0, b_start - 0.8, prev_end)
            display_end = words[-1]["end"] + 0.1
            prev_end = display_end

            s_str, e_str = self._convert_ssa_time(display_start), self._convert_ssa_time(display_end)
            is_cjk = bool(self.RE_CJK.search(line["text"]))
            style = "CJK_Duet" if is_cjk else "Lat_Duet"

            wait_cs = int(max(0, b_start - display_start) * 100)
            fade_in = max(0, min(300, int((b_start - display_start) * 1000)))
            k_tokens = []
            current_time = display_start
            for i, w in enumerate(words):
                gap_before = w['start'] - current_time
                if gap_before > 0.01: # placeholder to handle tiny floating point gap
                    if (gap_cs := int(round(gap_before * 100))) > 0: #ms
                        k_tokens.append(f"{{\\kf{gap_cs}}} ")
                word_cs = max(1, int(round((w['end'] - max(w['start'], current_time)) * 100)))
                k_tokens.append(f"{{\\kf{word_cs}}}{w['word']} ")
                current_time = w["end"]
            gap_after = display_end - current_time
            if gap_after > 0.01:
                gap_cs = int(round(gap_after * 100))
                if gap_cs > 0:
                    k_tokens.append(f"{{\\kf{gap_cs}}}")
            k_content = "".join(k_tokens)

            #k_content = " ".join([f"{{\\kf{max(1, int((w['end'] - w['start']) * 100))}}}{w['word']}" for w in words])
            main_ev = f"Dialogue: 0,{s_str},{e_str},{style},,20,20,50,,{{\\fad({fade_in},200)}}{{\\kf{wait_cs}}}{{\\an2}}{k_content}"
            #main_ev = f"Dialogue: 0,{s_str},{e_str},{style},,20,20,50,,{{\\fad({fade_in},200)}}{{\\an2}}{k_content}"

            events.append(main_ev)

        return self.ASS_HEADER + "\n".join(events) + "\n"
    
    def render(self, output_path: Optional[str], title: str, video_path: str, inst_path: str, duration: float, karaoke_data):
        safe_title = "".join(c for c in title if c.isalnum() or c in ' -_').rstrip()
        if output_path is None:
            output_path = f"{safe_title} (Karaoke).mkv"
        if self.with_ass:
            ass_content = self.build_ass(karaoke_data)
            ass_path = Path(config.work_dir) / f"{safe_title}_karaoke.ass"
            ass_path.write_text(ass_content, encoding="utf-8")
            ass_esc = str(ass_path).replace("\\", "/").replace(":", r"\\:")
            filters = f"[0:v]{self.scale_filter}[base];[base]ass=filename={ass_esc}[subbed];[subbed]copy[outv]"
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(inst_path),
                   "-filter_complex", filters,
                   "-map", "[outv]", "-map", "1:a:0", "-map", "0:a:0",
                   "-metadata", f"title={safe_title} (Karaoke+ by iantirta Version)", "-metadata", "artist=kplus",
                   "-metadata:s:a:0", "title=Karaoke", "-disposition:a:0", "default",
                   "-metadata:s:a:1", "title=Original Song", "-disposition:a:1", "none",
                   "-t", str(duration),
                   "-c:v", "h264_nvenc", "-cq", "34", "-preset", "p1", "-tune", "hq", "-rc", "vbr", "-b:v", "0", "-maxrate", "4M", "-bufsize", "8M",
                   "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(inst_path),
                   "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0",
                   "-metadata", f"title={safe_title} (Karaoke+ by iantirta Version)", "-metadata", "artist=kplus",
                   "-metadata:s:a:0", "title=Karaoke", "-disposition:a:0", "default",
                   "-metadata:s:a:1", "title=Original Song", "-disposition:a:1", "none",
                   "-t", str(duration),
                   "-c:v", "copy",  # <--- THIS IS THE MAGIC SPEED BULLET
                   "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path)]
        logger.info(">> Rendering video...")
        stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            stdout = sys.stdout
            stderr = sys.stderr
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)
        logger.info(f">> Successfully rendered: {output_path}")
        return output_path