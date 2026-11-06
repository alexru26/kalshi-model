"""The live GUI, rendered headless and driven through its widgets as a user would."""

import re
import time
from dataclasses import fields

import ipyvuetify as v
import plotly.graph_objects as go
import pytest
import solara

from kalshi_model import gui, live
from kalshi_model.config import Config
from kalshi_model.model import MarketModel
from kalshi_model.traits import TRAIT_NAMES


def render():
    _box, rc = solara.render(solara.AppLayout(children=[gui.Page()]), handle_error=False)
    return rc


@pytest.fixture
def page():
    rc = render()
    yield rc
    rc.close()


def status(rc) -> str:
    return rc.find(v.Html, tag="span").widget.children[0]


def step_count(rc) -> int:
    return int(status(rc).split()[1])


def click(rc, label: str) -> None:
    rc.find(v.Btn, children=[label]).widget.click()


def type_into(rc, label: str, text: str) -> None:
    """Type into a field and leave it, which is when the GUI applies an edit."""
    field = rc.find(v.TextField, label=label).widget
    field.v_model = text
    field.fire_event("focusout", None)


def choose(rc, label: str, value: str) -> None:
    rc.find(v.Select, label=label).widget.v_model = value


def errors(rc) -> list[str]:
    return [w.children[0] for w in rc.find(v.Alert, type="error").widgets]


def charts(rc) -> list[str]:
    """The titles of the charts shown on the page, in order (hidden ones left out)."""
    shown = []

    def walk(widget, hidden: bool) -> None:
        if isinstance(widget, go.FigureWidget):
            if not hidden:
                shown.append(widget.layout.title.text)
            return
        style = str(getattr(widget, "style_", "") or "").replace(" ", "")
        hidden = hidden or "display:none" in style
        for child in getattr(widget, "children", ()) or ():
            if not isinstance(child, str):
                walk(child, hidden)

    walk(rc.find(v.Html).widget, False)
    return shown


def short(title: str) -> str:
    """A chart title up to its first colon or line break."""
    return re.split(r":|<br>", title)[0]


def test_starts_paused_at_step_0_of_the_default(page):
    assert status(page) == f"Step 0 of {Config().n_steps} (paused)"
    assert page.find(v.Select, label="Preset").widget.v_model == live.DEFAULT_PRESET
    assert [short(title) for title in charts(page)] == [
        "Price vs P*",
        "Volume per step",
        "Bid, ask, and spread",
        "Who knows the latest clue",
        "Depth and market maker",
    ]


def test_editor_shows_every_config_field_and_trait(page):
    labels = {w.label for w in page.find(v.TextField).widgets}
    labels |= {w.label for w in page.find(v.Select).widgets}
    assert {f.name for f in fields(Config)} <= labels
    pins = {label for label in labels if label.startswith("pin: ")}
    assert pins == {"pin: n_clues", "pin: cash", "pin: activity"}
    # Every other trait is a Config field, edited once under Agent traits.
    assert set(TRAIT_NAMES) - {"n_clues", "cash", "activity"} <= labels


def test_step_advances_one_render_interval(page):
    click(page, "Step")
    assert step_count(page) == 1
    page.find(v.Slider, label="Render Interval: 1 steps").widget.v_model = 5
    click(page, "Step")
    assert step_count(page) == 6


def _wait_for(condition, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_play_runs_until_paused(page):
    page.find(v.Slider, label="Play Interval: 100 ms").widget.v_model = 1
    click(page, "▶")
    _wait_for(lambda: step_count(page) >= 3)
    click(page, "❚❚")
    _wait_for(lambda: "paused" in status(page))
    time.sleep(0.3)
    paused_at = step_count(page)
    time.sleep(0.5)
    assert step_count(page) == paused_at


def test_pause_stops_a_batch_and_step_never_overlaps_it(monkeypatch):
    calls, active, overlaps = [0], [0], [0]
    real_step = MarketModel.step

    def step(model):
        active[0] += 1
        overlaps[0] += active[0] > 1
        try:
            real_step(model)
        finally:
            active[0] -= 1
            calls[0] += 1

    monkeypatch.setattr(MarketModel, "step", step)
    page = render()
    try:
        page.find(v.Slider, label="Play Interval: 100 ms").widget.v_model = 1
        page.find(v.Slider, label="Render Interval: 1 steps").widget.v_model = 100
        click(page, "▶")
        _wait_for(lambda: calls[0] >= 150)
        click(page, "❚❚")
        at_pause = calls[0]
        click(page, "Step")
        assert 100 <= calls[0] - at_pause <= 101
        time.sleep(0.3)
        assert step_count(page) == calls[0]
        assert overlaps[0] == 0
    finally:
        page.close()


def test_play_stops_at_settlement(page):
    choose(page, "Preset", "Level 0")
    page.find(v.Slider, label="Play Interval: 100 ms").widget.v_model = 1
    page.find(v.Slider, label="Render Interval: 1 steps").widget.v_model = 100
    click(page, "▶")
    # The play loop resets the button just after the last step.
    _wait_for(lambda: page.find(v.Btn, children=["▶"]).widgets)
    assert status(page) == "Step 500 of 500 (settled)"
    assert page.find(v.Btn, children=["▶"]).widget.disabled
    assert page.find(v.Btn, children=["Step"]).widget.disabled


def test_editing_a_field_restarts_the_run(page):
    click(page, "Step")
    type_into(page, "n_steps", "300")
    assert status(page) == "Step 0 of 300 (paused)"


def test_an_invalid_edit_shows_why_and_keeps_the_current_run(page):
    click(page, "Step")
    type_into(page, "turns_per_step", "400")
    assert errors(page) == ["turns_per_step cannot exceed n_agents"]
    assert step_count(page) == 1
    type_into(page, "turns_per_step", "10")
    assert errors(page) == []
    assert step_count(page) == 0


def test_choosing_a_preset_loads_all_its_settings(page):
    type_into(page, "n_agents", "200")
    choose(page, "Preset", "Level 1")
    assert status(page) == "Step 0 of 500 (paused)"
    assert page.find(v.TextField, label="n_agents").widget.v_model == "3"
    assert page.find(v.Select, label="network").widget.v_model == "watts_strogatz"
    assert page.find(v.TextField, label="pin: n_clues").widget.v_model == "5"


def test_an_invalid_pin_shows_why(page):
    type_into(page, "pin: n_clues", "[1, 2]")
    assert errors(page) == ["trait 'n_clues' has 2 values for 300 agents"]
    type_into(page, "pin: n_clues", "")
    assert errors(page) == []


def test_seed_change_restarts_the_run(page):
    click(page, "Step")
    type_into(page, "seed", "7")
    assert step_count(page) == 0


def figure_ids(rc) -> set[int]:
    return {id(w) for w in rc.find(go.FigureWidget).widgets}


@pytest.mark.parametrize(
    "rebuild",
    [
        lambda rc: click(rc, "Reset"),
        lambda rc: type_into(rc, "seed", "7"),
        lambda rc: choose(rc, "Preset", "E3: spoofing"),
        lambda rc: choose(rc, "Preset", "Level 2 (default)"),
    ],
    ids=["reset", "seed", "preset", "back-to-default"],
)
def test_a_rebuild_reuses_the_chart_widgets(page, rebuild):
    # Each new Plotly widget sends plotly.js (about 5 MB) to the browser, which made
    # every edit take seconds, so a rebuild must point the same widgets at the new run.
    choose(page, "Preset", "E3: pump and dump")
    click(page, "Step")
    before = figure_ids(page)
    rebuild(page)
    assert figure_ids(page) == before
    assert step_count(page) == 0
    price = next(
        w for w in page.find(go.FigureWidget).widgets if w.layout.title.text == "Price vs P*"
    )
    assert len(price.data[0].x) == 0  # the new run's empty traces, not the old run's


def test_amm_drops_the_order_book_charts(page):
    type_into(page, "taker_fee_bp", "0")
    choose(page, "mechanism", "amm")
    assert errors(page) == []
    assert [short(title) for title in charts(page)] == [
        "Price vs P*",
        "Volume per step",
        "Who knows the latest clue",
    ]
    info = page.find(v.Alert, type="info").widget.children[0]
    assert "no order book" in info


@pytest.mark.parametrize(
    ("preset", "extra"),
    [
        (
            "E3: honest twin",
            ["Manipulator: position and P&L (twin)", "Manipulator: its orders vs the book (twin)"],
        ),
        (
            "E3: pump and dump",
            [
                "Manipulator: position and P&L (pump)",
                "Manipulator: its orders vs the book (pump)",
                "Price vs the honest twin (pump)",
            ],
        ),
        (
            "E3: spoofing",
            [
                "Manipulator: position and P&L (spoof)",
                "Manipulator: its orders vs the book (spoof)",
                "Price vs the honest twin (spoof)",
            ],
        ),
    ],
)
def test_e3_presets_add_the_manipulator_charts(page, preset, extra):
    choose(page, "Preset", preset)
    assert sorted(charts(page)[5:]) == sorted(extra)


def test_charts_follow_the_run_as_it_steps(page):
    choose(page, "Preset", "E3: pump and dump")
    page.find(v.Slider, label="Render Interval: 1 steps").widget.v_model = 25
    for _ in range(2):
        click(page, "Step")
    figures = {w.layout.title.text: w for w in page.find(go.FigureWidget).widgets}
    price = figures["Price vs P*"]
    assert len(price.data[0].x) == 50
    assert price.data[0].x[-1] == 50
    twin = figures["Price vs the honest twin (pump)"]
    assert len(twin.data[0].x) == 50  # the twin steps alongside
    assert list(twin.data[2].y) == [0.0] * 50  # no manipulation before manip_start
