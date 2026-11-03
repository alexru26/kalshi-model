from pathlib import Path

import numpy as np
import pytest

from kalshi_model.config import Config, amm_config
from kalshi_model.custom import load_custom
from kalshi_model.model import MarketModel
from kalshi_model.network_plot import (
    agent_graph,
    color_limit,
    heard_fake,
    mean_points,
    money_summary,
    network_chart,
    private_clues,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def run(config: Config, seed: int = 3) -> MarketModel:
    model = MarketModel(config, seed)
    model.run()
    return model


@pytest.fixture(scope="module")
def kalshi_with_fakes():
    return run(Config(n_steps=600, disinfo_ratio=0.2))


def by_gid(fig, gid: str):
    (match,) = [c for ax in fig.axes for c in ax.get_children() if c.get_gid() == gid]
    return match


def test_agent_graph_matches_each_agents_neighbors(kalshi_with_fakes):
    model = kalshi_with_fakes
    graph = agent_graph(model)
    assert graph.number_of_nodes() == len(model.traders)
    assert graph.number_of_edges() == model.graph.number_of_edges()
    for agent, neighbors in enumerate(model.neighbors):
        assert sorted(graph.neighbors(agent)) == neighbors


def test_nodes_are_colored_by_each_agents_pnl(kalshi_with_fakes):
    model = kalshi_with_fakes
    fig = network_chart(model)
    nodes = by_gid(fig, "agents")
    pnl = np.array([t.pnl for t in model.traders]) / 100
    assert np.allclose(nodes.get_array(), pnl)
    assert len(by_gid(fig, "links").get_segments()) == model.graph.number_of_edges()
    # Node area grows with private clues.
    sizes, clues = nodes.get_sizes(), private_clues(model)
    assert np.all(np.diff(sizes[np.argsort(clues, kind="stable")]) >= 0)


def test_scatter_panels_plot_pnl_against_degree_and_clues(kalshi_with_fakes):
    model = kalshi_with_fakes
    fig = network_chart(model)
    pnl = np.array([t.pnl for t in model.traders]) / 100
    degree = np.array([len(n) for n in model.neighbors])
    for gid, x in (("degree", degree), ("clues", private_clues(model))):
        points = by_gid(fig, gid).get_offsets()
        assert np.allclose(points[:, 1], pnl)
        assert np.all(np.abs(points[:, 0] - x) <= 0.2)  # jittered by at most 0.2


def test_money_sums_to_zero_on_the_order_book(kalshi_with_fakes):
    money = money_summary(kalshi_with_fakes)
    assert set(money) == {"traders", "market maker", "fees"}
    assert sum(money.values()) == pytest.approx(0)


def test_subtitle_counts_fake_hearers_and_escapes_dollars(kalshi_with_fakes):
    model = kalshi_with_fakes
    subtitle = network_chart(model).texts[-1].get_text()
    assert f"{heard_fake(model).sum()} of {len(model.traders)} agents heard a fake" in subtitle
    assert "$" not in subtitle.replace("\\$", "")  # a bare $ would start math text


def test_amm_money_includes_the_market_maker():
    model = run(amm_config(n_steps=300))
    assert set(money_summary(model)) == {"traders", "market maker"}
    assert "heard a fake" not in network_chart(model).texts[-1].get_text()


def test_constant_panels_give_way_to_per_agent_bars():
    # No network, so every agent has degree 0, and no private clues.
    custom = load_custom(CONFIGS / "test_2.toml", {"n_steps": 200, "mean_degree": 0})
    model = MarketModel(custom.config, 0, fixed_traits=custom.fixed_traits)
    model.run()
    fig = network_chart(model)
    bars = [p for ax in fig.axes for p in ax.patches if p.get_gid() == "pnl_bars"]
    assert np.allclose([b.get_height() for b in bars], [t.pnl / 100 for t in model.traders])
    assert not any(c.get_gid() in ("degree", "clues") for ax in fig.axes for c in ax.collections)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert labels == [str(i) for i in range(len(model.traders))]


def test_mean_points_per_value_and_per_bin():
    x = np.array([0, 0, 1, 1, 2])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    mx, my = mean_points(x, y)
    assert mx.tolist() == [0, 1, 2] and my.tolist() == [2, 6, 9]
    x = np.arange(100)
    mx, my = mean_points(x, 2.0 * x)
    assert 1 < len(mx) <= 8
    assert np.allclose(my, 2 * mx)


def test_color_limit_ignores_outliers_only_in_large_runs():
    assert color_limit(np.array([1.0, -50.0])) == 50
    big = np.r_[np.linspace(-10, 10, 99), 1000.0]
    assert color_limit(big) < 20
    assert color_limit(np.zeros(5)) > 0


def test_unsettled_run_is_rejected():
    model = MarketModel(Config(n_steps=300), 0)
    model.step()
    with pytest.raises(ValueError):
        network_chart(model)


def test_triangle_config_is_a_triangle_of_equal_agents():
    custom = load_custom(CONFIGS / "test_3.toml", {"n_steps": 60})
    model = MarketModel(custom.config, 20000, fixed_traits=custom.fixed_traits)
    model.run()
    assert sorted(len(n) for n in model.neighbors) == [1, 1, 2]  # a line
    fig = network_chart(model)
    # Every agent holds the same clues: one node size, no size key, said in the subtitle.
    assert len(set(by_gid(fig, "agents").get_sizes())) == 1
    assert fig.axes[0].get_legend() is None
    assert "every agent dealt 3 private clues" in fig.texts[-1].get_text()
