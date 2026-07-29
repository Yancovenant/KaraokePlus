
from typing import TYPE_CHECKING

from kplus.environment import env

from .utils import AudioSegment, Result, Segment, WordTiming, sec2ass, _process_audio

if TYPE_CHECKING:
    import numpy as np  # type: ignore

    from .utils import AudioType

class Refiner:
    def __init__(self, sr: int, precision_ms: int):
        env.plotly
        self.sr = sr
        self.hop_length = int(self.sr / 1000) * precision_ms # 16 frames
        self.frame_length = int(self.hop_length * 2) # 32 frames
        import plotly.graph_objects as go  # type: ignore
        self.go = go

    def _split_audio(self, audio_np: np.ndarray, start: float, end: float):
        start_sample, end_sample = int(start*self.sr), int(end*self.sr)
        return audio_np[start_sample:end_sample]

    def _draw_waveform(self, fig, audio_chunk, safe_start, safe_end):
        time_axis = np.linspace(safe_start, safe_end, len(audio_chunk))
        fig.add_trace(self.go.Scatter(x=time_axis, y=audio_chunk, name="Waveform",
                                line={"color": "#00d2ff", "width": 1}), row=1, col=1)

    def _draw_rms(self, fig, audio_chunk, safe_start):
        env.librosa; import librosa  # type: ignore  # noqa: B018, I001
        rms = librosa.feature.rms(y=audio_chunk, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        rms_times = librosa.frames_to_time(np.arange(len(rms)),sr=self.sr, hop_length=self.hop_length) + safe_start
        fig.add_trace(self.go.Scatter(x=rms_times, y=rms_db, name="RMS (dB)",
                                fill="tozeroy", line={"color": "#ffaa00", "width": 2}), row=2, col=1)
    
    def _draw_res_seg(self, fig, seg: Segment, color, row):
        fig.add_vrect(x0=seg.start, x1=seg.end, fillcolor=color, opacity=0.3,
            layer="below", line_width=0, row=row)
        fig.add_vline(x=seg.start, line_width=2, line_color=color, opacity=0.7, name="WordBoundaries", row=row)
        fig.add_vline(x=seg.end, line_width=2, line_color=color, opacity=0.7, name="WordBoundaries", row=row)

    def refine_timestamp(self, audio: AudioType,
        sr: int, res_1: Result, res_2: Result, audio_segments: list[AudioSegment]
    ) -> Result:
        from plotly.subplots import make_subplots  # type: ignore
        audio_np = _process_audio(audio, sr, self.sr)
        for i, (seg1, seg2, audio_segment) in enumerate(zip(res_1.segments, res_2.segments, audio_segments)):
            if i > 0: break
            safe_start = min(seg1.start, seg2.start, audio_segment.start)
            safe_end = max(seg1.end, seg2.end, audio_segment.end)
            audio_chunk = self._split_audio(audio_np, safe_start, safe_end)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, subplot_titles=("Raw Waveform", "RMS Energy (dB)", "Seg1", "Seg2"))
            self._draw_waveform(fig, audio_chunk, safe_start, safe_end)
            self._draw_rms(fig, audio_chunk, safe_start)
            self._draw_res_seg(fig, seg1, "green", 3)
            self._draw_res_seg(fig, seg2, "red", 4)
            fig.update_layout(template="plotly_dark", hovermode="x unified",
                height=450, margin={"l": 20, "r": 20, "t": 40, "b": 20},showlegend=False
            )
            fig.show(config={"staticPlot": True})
            print("ok")
        print("done")