import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.collections import PathCollection

from kalshi_model.config import Config, amm_config
from kalshi_model.model import MarketModel
from kalshi_model.plot import clue_events, price_chart


def run(config: Config, seed: int = 3) -> MarketModel:
    model = MarketModel(config, seed)
    model.run()
    return model


@pytest.fixture(scope="module")
def kalshi_with_fakes():
    return run(Config(n_steps=600, disinfo_ratio=0.2))


def line(ax, label: str) -> np.ndarray:
    (match,) = [ln for ln in ax.get_lines() if ln.get_label() == label]
    return np.asarray(match.get_ydata(), dtype=float)


def markers(ax) -> dict[str, np.ndarray]:
    return {
        c.get_label(): c.get_offsets()[:, 0]
        for c in ax.collections
        if isinstance(c, PathCollection)
    }


def test_clue_events_list_every_released_clue_with_its_direction(kalshi_with_fakes):
    model = kalshi_with_fakes
    events = clue_events(model)
    assert [e.clue_id for e in events] == sorted(model.release_step, key=model.release_step.get)
    assert {e.clue_id for e in events if e.fake} == set(model.fake_ids) > set()
    for e in events:
        assert e.step == model.release_step[e.clue_id]
        assert e.points_yes == model.points_yes(e.clue_id)
    # Fakes always point against the outcome.
    assert all(e.points_yes != bool(model.world.y) for e in events if e.fake)


def test_price_panel_plots_the_recorded_price_and_p_star_in_cents(kalshi_with_fakes):
    model = kalshi_with_fakes
    df = model.datacollector.get_model_vars_dataframe()
    price_ax = price_chart(model).axes[0]
    assert np.allclose(line(price_ax, "price"), 100 * df.price, equal_nan=True)
    assert np.allclose(line(price_ax, "P* (benchmark)"), 100 * df.p_star)
    (price,) = [ln for ln in price_ax.get_lines() if ln.get_label() == "price"]
    assert price.get_xdata()[0] == 1  # row r is step r + 1


def test_each_released_clue_gets_one_marker_at_its_release_step(kalshi_with_fakes):
    model = kalshi_with_fakes
    events = clue_events(model)
    drawn = markers(price_chart(model).axes[0])
    for fake, kind in ((False, "news"), (True, "fake")):
        for yes, way in ((True, "YES"), (False, "NO")):
            expected = sorted(e.step for e in events if e.fake == fake and e.points_yes == yes)
            label = f"{kind} clue, points {way}"
            assert sorted(drawn.get(label, [])) == expected
    assert sum(len(v) for k, v in drawn.items() if "clue" in k) == len(events)


def test_order_book_chart_has_volume_and_spread_panels(kalshi_with_fakes):
    model = kalshi_with_fakes
    df = model.datacollector.get_model_vars_dataframe()
    fig = price_chart(model)
    _, volume_ax, spread_ax = fig.axes
    assert sum(p.get_height() for p in volume_ax.patches) == df.volume.sum()
    assert np.allclose(
        spread_ax.get_lines()[0].get_ydata(), df.spread.astype(float), equal_nan=True
    )
    empty = markers(spread_ax).get("one side of the book empty", [])
    assert len(empty) == df.empty_book.sum()


def test_amm_chart_has_no_spread_panel_and_no_fake_markers():
    model = run(amm_config(n_steps=400))
    fig = price_chart(model)
    assert len(fig.axes) == 2
    assert not any("fake" in label for label in markers(fig.axes[0]))


def test_the_outcome_is_labeled(kalshi_with_fakes):
    model = kalshi_with_fakes
    texts = [t.get_text() for t in price_chart(model).axes[0].texts]
    outcome = "YES" if model.world.y else "NO"
    assert any(f"resolved {outcome}" in t and f"{100 * model.price:.1f}c" in t for t in texts)


def test_unsettled_run_is_rejected():
    model = MarketModel(Config(n_steps=300), 0)
    model.step()
    with pytest.raises(ValueError):
        price_chart(model)


def test_plot_script_writes_a_png(tmp_path):
    out = tmp_path / "run.png"
    script = Path(__file__).resolve().parents[1] / "experiments" / "plot_run.py"
    done = subprocess.run(
        [sys.executable, str(script), "5", "--market", "amm", "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(out) in done.stdout
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
