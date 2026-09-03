import json
import logging
from pathlib import Path

from kplus import env
from kplus.tools import rich

logger = logging.getLogger(__name__)

class AudioPlotter:
    def __init__(self, *args, **plot_kwargs):
        env.plotly  # noqa: B018
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
        self.go = go
        self.make_subplots = make_subplots
        self._plot_kwargs = {
            "shared_xaxes": True,
            "vertical_spacing": 0.05,
        }
        valid_plot_kwargs = {k: v for k, v in plot_kwargs.items() if k in self._plot_kwargs}
        self._plot_kwargs.update(valid_plot_kwargs)
        self.refresh()
        from plotly.io import renderers
        self.has_renderer = renderers.default.strip()
        if plot_kwargs.pop("resample", False):
            env.plotly_resampler  # noqa: B018
            from plotly_resampler import register_plotly_resampler  # type: ignore
            register_plotly_resampler(mode='auto')
        self.use_html = plot_kwargs.pop("use_html", False)
        if env.is_colab:
            from google.colab import output
            output.enable_custom_widget_manager()
            logger.debug("Google colab enabled custom widget manager")

    def hex2rgba(self, h: str, op: float) -> str:
        n = int(h.lstrip('#').ljust(8, 'F'), 16)
        a = op if op is not None else round((n & 255) / 255, 2)
        return f"rgba{(n >> 24, (n >> 16) & 255, (n >> 8) & 255, a)}"

    def scatter(self, func_name: str = "Scatter", row=1, col=1, **data):
        self._col = max(col, self._col)
        self._row = max(row, self._row)
        self._scatter.append((
            getattr(self.go, func_name)(**data), row, col
        ))

    def update_titles(self, titles:list[str]):
        self._titles = list(titles)

    def refresh(self):
        self._row, self._col = 1, 1
        self._scatter: list[tuple] = []
        self._titles: list[str] = [" "]
        self._layout_kw: list[dict] = []

    def update_layout(self, **kwargs):
        self._layout_kw.append(kwargs)

    def show(self, *, audio_uri=None, segments=None, **kwargs):
        fig = self.make_subplots(
            rows=self._row, cols=self._col, subplot_titles=self._titles,
            **self._plot_kwargs
        )
        fig.add_traces(
            data=[s[0] for s in self._scatter],
            rows=[s[1] for s in self._scatter],
            cols=[s[2] for s in self._scatter],
        )

        dynamic_height = 180 * self._row
        for kw in self._layout_kw:
            fig.update_layout(**kw)
        fig.update_layout(template="plotly_dark", hovermode="x unified",
            height=dynamic_height, margin={"l": 20, "r": 20, "t": 40, "b": 20},
            showlegend=False,
        )
        if self.has_renderer and not self.use_html:
            fig.show(**kwargs)
        else:
            outpath = "test.html"
            if audio_uri is not None and segments is not None:
                segment_data = []
                for i, segment in enumerate(segments):
                    segment_data.append({
                        "index": i,
                        "start": float(segment.start),
                        "end": float(segment.end),
                    })
                segment_json = json.dumps(segment_data)
                fightml = fig.to_html(
                    full_html=False,
                    include_plotlyjs=True,
                    config={
                        "responsive": True,
                        "displaylogo": False,
                    },
                )
                from gettext import gettext as _  # noqa: I001
                from .plotter_utils import audio_html, base_html
                content = fightml
                content += _(audio_html) % dict(segment_json=segment_json, audio_uri=audio_uri)
                fullhtml = _(base_html) % dict(content=content)
                with open(outpath, "w", encoding="utf-8") as f:
                    f.write(fullhtml)
            else:
                fig.write_html(outpath)
            rich.print("file://" + str(Path(outpath).expanduser().resolve()))
            