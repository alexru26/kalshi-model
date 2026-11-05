"""A live GUI: edit the model's settings and watch one run play out.

Launch it with `uv run solara run kalshi_model.gui` and open the address it prints
(http://localhost:8765 by default; from WSL, open it in the Windows browser).

The sidebar holds the run controls, laid out like Mesa's SolaraViz: Reset, play/pause,
and Step, a play interval (milliseconds between updates), and a render interval (steps
per update). Below them are the presets (`live.PRESETS`), the seed, every `Config` field
grouped by section, and a pin for each agent trait. Any change rebuilds the run at step
0 with the new settings and pauses it; an invalid setting shows its error and leaves
the current run as it was.

The main area shows Plotly charts (`live_charts`), each built once per run and then fed
new data in place: the price against P*, volume, the bid, ask, and spread, book depth
and the market maker, and the network colored by who knows the latest clue. Order book
charts need the order book, so the legacy AMM shows only the price, volume, and network.
E3 runs add the manipulator's position and P&L, its orders against the book, and, for
pump and dump or spoofing, its price against the honest twin's on the same seed (the
twin runs alongside, in step).
"""

from __future__ import annotations

import threading
import time

import ipyvuetify as v
import solara
import solara.lab
import traitlets

from kalshi_model import live
from kalshi_model.live_charts import (
    MAIN_CHARTS,
    MANIPULATOR_CHARTS,
    Chart,
    Run,
    apply,
    reload,
    widget,
)
from kalshi_model.model import MarketModel
from kalshi_model.traits import TRAIT_NAMES

PLAY_INTERVAL = (1, 500, 10, 100)
"""Play interval slider in milliseconds: min, max, step, start (Mesa's SolaraViz values)."""
RENDER_INTERVAL = (1, 100, 2, 1)
"""Render interval slider in steps: min, max, step, start (Mesa's SolaraViz values)."""


def make_run(values: dict[str, object], pins: dict[str, object], seed: int) -> Run:
    """A fresh run at step 0, with an honest twin on the same seed when the manipulator
    pumps and dumps or spoofs. Raises ValueError for any invalid setting or pin."""
    model = live.build_model(values, pins, seed)
    twin = None
    if model.config.manipulator in ("pump", "spoof"):
        twin = live.build_model({**values, "manipulator": "twin"}, pins, seed)
    return Run(model, twin, live.network_layout_for(model))


class Session:
    """One browser tab's state: the settings, the run, and the playback controls.

    Settings live in reactive variables so every input and chart rerenders when they
    change. `run` holds the model, its honest twin (pump and dump or spoofing only),
    and the network layout. `version` counts chart updates: the play loop bumps it
    after each batch of steps, since stepping the model mutates it in place. `lock` lets
    one thread at a time step the run or read it for the charts, since the play loop
    steps it in a worker thread.
    """

    def __init__(self) -> None:
        preset = live.PRESETS[live.DEFAULT_PRESET]
        self.preset = solara.reactive(live.DEFAULT_PRESET)
        self.values = solara.reactive(dict(preset.values))
        self.pin_text = solara.reactive(
            {name: live.format_pin(v) for name, v in preset.pins.items()}
        )
        self.seed = solara.reactive(live.DEFAULT_SEED)
        self.error = solara.reactive(None)
        self.run = solara.reactive(make_run(preset.values, preset.pins, self.seed.value))
        self.playing = solara.reactive(False)
        self.play_interval = solara.reactive(PLAY_INTERVAL[3])
        self.render_interval = solara.reactive(RENDER_INTERVAL[3])
        self.version = solara.reactive(0)
        self.lock = threading.Lock()
        self.figures: dict[Chart, object] = {}
        """One Plotly widget per chart, made the first time the chart applies and then
        kept (reloaded for each new run), so plotly.js reaches the browser once."""
        self._loaded: dict[Chart, Run] = {}

    def figure(self, chart: Chart, run: Run):
        """The chart's widget, laid out for `run`."""
        with self.lock:
            if chart not in self.figures:
                self.figures[chart] = widget(chart, run)
            elif self._loaded[chart] is not run:
                reload(self.figures[chart], chart, run)
            self._loaded[chart] = run
        return self.figures[chart]

    def rebuild(
        self,
        values: dict[str, object] | None = None,
        pin_text: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Apply new settings: start a fresh, paused run, or show why they are invalid.

        The inputs keep what was typed either way, so an error can be fixed in place.
        """
        values = self.values.value if values is None else values
        pin_text = self.pin_text.value if pin_text is None else pin_text
        seed = self.seed.value if seed is None else seed
        self.values.value, self.pin_text.value, self.seed.value = values, pin_text, seed
        self.playing.value = False
        try:
            pins = {}
            for name, text in pin_text.items():
                value = live.parse_pin(name, text)
                if value is not None:
                    pins[name] = value
            run = make_run(values, pins, seed)
        except ValueError as e:
            self.error.value = str(e)
            return
        self.error.value = None
        self.run.value = run
        self.version.value += 1

    def apply_preset(self, name: str) -> None:
        preset = live.PRESETS[name]
        self.preset.value = name
        pins = {trait: live.format_pin(v) for trait, v in preset.pins.items()}
        self.rebuild(dict(preset.values), pins)

    def set_value(self, name: str, value: object) -> None:
        self.rebuild({**self.values.value, name: value})

    def set_pin(self, name: str, text: str) -> None:
        self.rebuild(pin_text={**self.pin_text.value, name: text})

    def step(self, n: int, while_playing: bool = False) -> None:
        """Advance the run up to n steps (stopping at settlement, or at a pause when
        `while_playing`) and redraw once."""
        with self.lock:
            run = self.run.value
            for _ in range(n):
                if run.model.settled or (while_playing and not self.playing.value):
                    break
                run.model.step()
                if run.twin is not None:
                    run.twin.step()
        self.version.value += 1

    def play(self) -> None:
        """The play loop: one batch of `render_interval` steps per `play_interval`, until
        paused, settled, or replaced by a rebuilt run."""
        run = self.run.value
        while self.playing.value and self.run.value is run and not run.model.settled:
            time.sleep(self.play_interval.value / 1000)
            if not self.playing.value or self.run.value is not run:
                return
            self.step(self.render_interval.value, while_playing=True)
        if self.run.value is run and run.model.settled:
            self.playing.value = False


@solara.component
def Controls(session: Session):
    model: MarketModel = session.run.value.model
    session.version.use_value()  # the step count changes as the run plays
    playing = session.playing.value
    solara.lab.use_task(
        session.play, dependencies=[playing, session.run.value], prefer_threaded=True
    )
    with solara.Card("Controls"):
        with solara.Row(justify="space-between"):
            solara.Button("Reset", color="primary", on_click=lambda: session.rebuild())
            solara.Button(
                "❚❚" if playing else "▶",
                color="primary",
                disabled=model.settled,
                on_click=lambda: session.playing.set(not session.playing.value),
            )
            solara.Button(
                "Step",
                color="primary",
                disabled=playing or model.settled,
                on_click=lambda: session.step(session.render_interval.value),
            )
        low, high, step, _ = PLAY_INTERVAL
        solara.SliderInt(
            f"Play Interval: {session.play_interval.value} ms",
            value=session.play_interval,
            min=low,
            max=high,
            step=step,
        )
        low, high, step, _ = RENDER_INTERVAL
        solara.SliderInt(
            f"Render Interval: {session.render_interval.value} steps",
            value=session.render_interval,
            min=low,
            max=high,
            step=step,
        )
        status = "settled" if model.settled else ("playing" if playing else "paused")
        solara.Text(f"Step {model.steps} of {model.config.n_steps} ({status})")


@solara.component
def FieldInput(session: Session, name: str):
    value = session.values.value[name]
    kind = live.FIELD_TYPES[name]

    def on_value(new: object) -> None:
        if new is not None and new != value:
            session.set_value(name, new)

    if name in live.CHOICES:
        solara.Select(name, value=value, values=list(live.CHOICES[name]), on_value=on_value)
    elif kind is int:
        solara.InputInt(name, value=value, on_value=on_value)
    else:
        solara.InputFloat(name, value=value, on_value=on_value)


@solara.component
def ErrorSlot(session: Session):
    """Why the latest settings were rejected. Always rendered, even when empty, so an
    error appearing never shifts (and so collapses) the sections below it."""
    with solara.Column():
        if session.error.value:
            solara.Error(session.error.value)


@solara.component
def Settings(session: Session):
    with solara.Card("Settings"):
        solara.Select(
            "Preset",
            value=session.preset.value,
            values=list(live.PRESETS),
            on_value=session.apply_preset,
        )
        solara.InputInt(
            "seed",
            value=session.seed.value,
            on_value=lambda s: session.rebuild(seed=s) if s != session.seed.value else None,
        )
        ErrorSlot(session)
        for title, names in live.FIELD_GROUPS:
            with solara.Details(title):
                for name in names:
                    FieldInput(session, name)
        with solara.Details("Pinned traits"):
            solara.Markdown(
                "Blank leaves a trait drawn from its distribution (n_clues, cash) or at "
                "its config value. A number pins it for every agent; a list such as "
                "`[20, 0]` gives one value per agent."
            )
            for name in TRAIT_NAMES:
                solara.InputText(
                    f"pin: {name}",
                    value=session.pin_text.value.get(name, ""),
                    on_value=lambda text, name=name: session.set_pin(name, text),
                )


class ResizeWatcher(v.VuetifyTemplate):
    """Browser-side helper: when the element around a chart changes width (as it does
    once the page lays out), fire a window resize, the one signal on which Plotly's
    widget refits itself to its container. Without it a chart keeps Plotly's default
    700 px width and is cut off in a narrower column."""

    template = traitlets.Unicode("""
<template><span style="display: none"></span></template>
<script>
module.exports = {
  mounted() {
    let timer = null;
    this.observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    });
    this.observer.observe(this.$el.parentElement);
  },
  destroyed() {
    this.observer.disconnect();
  },
};
</script>
""").tag(sync=True)


@solara.component
def LiveChart(session: Session, chart: Chart):
    """One chart: its session-long Plotly widget, fed new data on every update, and
    hidden while the run has nothing for it (no order book, no manipulator, no twin)."""
    run = session.run.value
    version = session.version.value
    shown = chart.applies(run)
    figure = session.figure(chart, run) if shown else session.figures.get(chart)

    def refresh() -> None:
        if not shown:
            return
        with session.lock:
            update = chart.update(run)
        apply(figure, update)

    solara.use_effect(refresh, [version, run])
    if figure is None:  # never shown yet
        return
    # A plain div: solara.Column's `d-flex !important` would override display: none.
    solara.Div(
        children=[figure, ResizeWatcher.element()],
        style={"width": "100%"} if shown else {"display": "none"},
    )


MANIPULATOR_COLUMNS = (
    (MANIPULATOR_CHARTS[0], MANIPULATOR_CHARTS[1]),
    (MANIPULATOR_CHARTS[2],),
)
"""Position and P&L above the twin chart on the left, the orders chart on the right."""


@solara.component
def Charts(session: Session):
    # Every chart stays in the page, hidden when it does not apply, so its widget is
    # never torn down and rebuilt.
    run = session.run.value
    for columns in (MAIN_CHARTS, MANIPULATOR_COLUMNS):
        with solara.Columns([1, 1]):
            for column in columns:
                with solara.Column():
                    for chart in column:
                        LiveChart(session, chart)
    if run.model.book is None:
        solara.Info("The legacy AMM has no order book: no bid, ask, or depth charts.")


@solara.component
def Page():
    session = solara.use_memo(Session, [])
    solara.Title("Kalshi model")
    with solara.AppBar():
        solara.AppBarTitle("Kalshi model: live run")
    with solara.Sidebar():
        Controls(session)
        Settings(session)
    Charts(session)
