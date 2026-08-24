
from kplus import env


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
        self._plot_kwargs.update(plot_kwargs)
        self.refresh()

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

    def show(self, *kwargs):
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
            height=dynamic_height, margin={"l": 20, "r": 20, "t": 40, "b": 20},showlegend=False,
        )
        fig.show()