"""A live GUI: edit the model's settings and watch one run play out.

Launch it with `uv run solara run kalshi_model.gui` and open the address it prints
(http://localhost:8765 by default; from WSL, open it in the Windows browser).

The sidebar holds the run controls, laid out like Mesa's SolaraViz: Reset, play/pause,
and Step, a play interval (milliseconds between updates), and a render interval (steps
per update). Below them are the presets (`live.PRESETS`), the seed, every `Config` field
grouped by section, and a pin for each agent trait. Any change rebuilds the run at step
0 with the new settings and pauses it; an invalid setting shows its error and leaves
the current run as it was.

The main area shows the price against P*, volume, the bid, ask, and spread, book depth
and the market maker, and the network colored by who knows the latest clue. Order book
charts need the order book, so the legacy AMM shows only the price, volume, and network.
"""

from __future__ import annotations

import threading
import time

import solara
import solara.lab

from kalshi_model import live
from kalshi_model.model import MarketModel
from kalshi_model.traits import TRAIT_NAMES

PLAY_INTERVAL = (1, 500, 10, 100)
"""Play interval slider in milliseconds: min, max, step, start (Mesa's SolaraViz values)."""
RENDER_INTERVAL = (1, 100, 2, 1)
"""Render interval slider in steps: min, max, step, start (Mesa's SolaraViz values)."""


class Session:
    """One browser tab's state: the settings, the run, and the playback controls.

    Settings live in reactive variables so every input and chart rerenders when they
    change. `version` counts chart updates: the play loop bumps it after each batch of
    steps, since stepping the model mutates it in place. `lock` lets one thread at a
    time step the run or draw it, since the play loop steps it in a worker thread.
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
        self.model = solara.reactive(live.build_model(preset.values, preset.pins, self.seed.value))
        self.layout = solara.reactive(live.network_layout_for(self.model.value))
        self.playing = solara.reactive(False)
        self.play_interval = solara.reactive(PLAY_INTERVAL[3])
        self.render_interval = solara.reactive(RENDER_INTERVAL[3])
        self.version = solara.reactive(0)
        self.lock = threading.Lock()

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
            model = live.build_model(values, pins, seed)
        except ValueError as e:
            self.error.value = str(e)
            return
        self.error.value = None
        self.layout.value = live.network_layout_for(model)
        self.model.value = model
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
            model = self.model.value
            for _ in range(n):
                if model.settled or (while_playing and not self.playing.value):
                    break
                model.step()
        self.version.value += 1

    def play(self) -> None:
        """The play loop: one batch of `render_interval` steps per `play_interval`, until
        paused, settled, or replaced by a rebuilt run."""
        model = self.model.value
        while self.playing.value and self.model.value is model and not model.settled:
            time.sleep(self.play_interval.value / 1000)
            if not self.playing.value or self.model.value is not model:
                return
            self.step(self.render_interval.value, while_playing=True)
        if self.model.value is model and model.settled:
            self.playing.value = False


@solara.component
def Controls(session: Session):
    model: MarketModel = session.model.value
    session.version.use_value()  # the step count changes as the run plays
    playing = session.playing.value
    solara.lab.use_task(
        session.play, dependencies=[playing, session.model.value], prefer_threaded=True
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


@solara.component
def Chart(session: Session, draw):
    """One live chart: redrawn whenever the run steps or is replaced."""
    version = session.version.value
    model = session.model.value
    with session.lock:
        fig = draw(model)
    if fig is None:
        return
    solara.FigureMatplotlib(fig, format="png", dependencies=[version, id(model)])


@solara.component
def Charts(session: Session):
    layout = session.layout.value

    def network(model):
        return live.network_figure(model, layout)

    has_book = session.model.value.book is not None
    with solara.Columns([1, 1]):
        with solara.Column():
            Chart(session, live.price_figure)
            Chart(session, live.volume_figure)
            if has_book:
                Chart(session, live.book_figure)
        with solara.Column():
            Chart(session, network)
            if has_book:
                Chart(session, live.market_maker_figure)
            else:
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
