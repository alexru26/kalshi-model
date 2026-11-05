"""The live GUI, rendered headless and driven through its widgets as a user would."""

import time
from dataclasses import fields

import ipyvuetify as v
import ipywidgets
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


def charts(rc) -> int:
    return len(rc.find(ipywidgets.Image).widgets)


def test_starts_paused_at_step_0_of_the_default(page):
    assert status(page) == f"Step 0 of {Config().n_steps} (paused)"
    assert page.find(v.Select, label="Preset").widget.v_model == live.DEFAULT_PRESET
    assert charts(page) == 5


def test_editor_shows_every_config_field_and_trait(page):
    labels = {w.label for w in page.find(v.TextField).widgets}
    labels |= {w.label for w in page.find(v.Select).widgets}
    assert {f.name for f in fields(Config)} <= labels
    assert {f"pin: {name}" for name in TRAIT_NAMES} <= labels


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


def test_amm_drops_the_order_book_charts(page):
    type_into(page, "taker_fee_bp", "0")
    choose(page, "mechanism", "amm")
    assert errors(page) == []
    assert charts(page) == 3  # price, volume, network
    info = page.find(v.Alert, type="info").widget.children[0]
    assert "no order book" in info
