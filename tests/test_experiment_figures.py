import numpy as np
import pytest

from kalshi_model.experiment import RESULTS_DIR, load, paired
from kalshi_model.experiment_figures import FIGURES, e2_distortion, write_figures


@pytest.fixture(scope="module")
def figures(tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    return out, write_figures(RESULTS_DIR, out, dpi=40)


def test_every_figure_is_drawn_from_the_saved_results(figures):
    out, written = figures
    assert sorted(p.name for p in written) == sorted(f"{name}.png" for name in FIGURES)
    for path in written:
        assert path.parent == out
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_distortion_figure_plots_the_paired_distortion_by_ratio():
    results = load("e2_information")
    fig = e2_distortion(results)
    whole = fig.axes[0].lines[0].get_ydata()
    ratios = sorted(float(label.split()[-1]) for label in results)[1:]
    expected = [paired(results, "signed_gap", "ratio 0.0", f"ratio {r}")[0] for r in ratios]
    assert np.allclose(whole, expected)
