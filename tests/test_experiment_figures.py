import re

import numpy as np
import pytest

from kalshi_model.experiment import RESULTS_DIR, load, paired
from kalshi_model.experiment_figures import (
    E1_PANELS,
    FIGURES,
    e1_rules,
    e2_distortion,
    write_figures,
)


@pytest.fixture(scope="module")
def figures(tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    return out, write_figures(RESULTS_DIR, out, dpi=40)


@pytest.fixture(scope="module")
def drawn():
    cache = {}
    out = {}
    for name, (experiment, draw) in FIGURES.items():
        if experiment not in cache:
            cache[experiment] = load(experiment)
        out[name] = draw(cache[experiment])
    return out


def test_every_figure_is_drawn_from_the_saved_results(figures):
    out, written = figures
    assert sorted(p.name for p in written) == sorted(f"{name}.png" for name in FIGURES)
    for path in written:
        assert path.parent == out
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_have_a_capitalized_title_and_no_subtitle(drawn):
    experiment_code = re.compile(r"\bE\d")
    for name, fig in drawn.items():
        title = fig.get_suptitle()
        assert [t.get_text() for t in fig.texts] == [title], name
        words = [w for w in title.split() if w not in {"and", "from", "of", "vs"}]
        assert all(w[0].isupper() for w in words), title
        panel_titles = [ax.get_title(loc="left") for ax in fig.axes]
        for text in [title, *panel_titles]:
            assert not experiment_code.search(text), text
        assert all(t[:1].isupper() for t in panel_titles if t), panel_titles


def test_distortion_figure_plots_the_paired_distortion_by_ratio():
    results = load("e2_information")
    fig = e2_distortion(results)
    means = fig.axes[0].lines[0].get_ydata()
    ratios = sorted(float(label.split()[-1]) for label in results)[1:]
    expected = [paired(results, "signed_gap", "ratio 0.0", f"ratio {r}")[0] for r in ratios]
    assert np.allclose(means, expected)


def test_fee_figure_states_each_paired_difference():
    results = load("e1_rules")
    fig = e1_rules(results)
    for ax, (_, metric, _) in zip(fig.axes, E1_PANELS, strict=True):
        if not isinstance(metric, str):
            continue
        (note,) = [t.get_text() for t in ax.texts if t.get_text().startswith("Kalshi - CLOB")]
        stated = float(re.search(r"Kalshi - CLOB: ([+-][\d.,]+)", note)[1].replace(",", ""))
        expected = paired(results, metric, "clob", "kalshi")[0]
        assert stated == pytest.approx(expected, abs=0.051)
