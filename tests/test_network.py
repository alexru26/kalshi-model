from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from kalshi_model.config import Config, amm_config
from kalshi_model.custom import load_custom
from kalshi_model.model import MarketModel
from kalshi_model.network import build_network, place_agents
from kalshi_model.rng import make_streams


def graph_for(seed: int = 0, **overrides) -> nx.Graph:
    return build_network(Config(**overrides), make_streams(seed)["network"])


def mean_degree(graph: nx.Graph) -> float:
    return 2 * graph.number_of_edges() / graph.number_of_nodes()


def clustering(seed: int, **overrides) -> float:
    return nx.average_clustering(graph_for(seed, **overrides))


# Network


def test_network_has_n_nodes_and_the_configured_mean_degree():
    degrees = [mean_degree(graph_for(seed)) for seed in range(5)]
    assert all(graph_for(s).number_of_nodes() == 300 for s in range(2))
    assert np.mean(degrees) == pytest.approx(8, abs=0.4)
    assert nx.number_of_selfloops(graph_for()) == 0


def test_same_seed_gives_the_same_network():
    assert nx.utils.graphs_equal(graph_for(3), graph_for(3))
    assert not nx.utils.graphs_equal(graph_for(3), graph_for(4))


def test_network_has_hubs_and_clustering():
    # Hubs: preferential attachment gives top degrees far above the mean of 8.
    assert np.mean([max(d for _, d in graph_for(s).degree()) for s in range(3)]) > 30
    # Clustering comes from triad closure: turning it off removes most of it.
    with_triads = np.mean([clustering(s) for s in range(3)])
    without = np.mean([clustering(s, hk_triad=0.0) for s in range(3)])
    assert with_triads > 2 * without


def test_zero_degree_means_no_network():
    graph = build_network(Config(mean_degree=0), make_streams(0)["network"])
    assert graph.number_of_nodes() == 300 and graph.number_of_edges() == 0


def test_level_1_demo_is_a_triangle():
    # The one non-Holme-Kim network: a Watts-Strogatz ring of 3, everyone linked.
    custom = load_custom(Path(__file__).resolve().parents[1] / "configs" / "level_1.toml")
    model = MarketModel(custom.config, 0, fixed_traits=custom.fixed_traits)
    assert model.neighbors == [[1, 2], [0, 2], [0, 1]]


@pytest.mark.parametrize(
    "bad",
    [{"network": "sbm"}, {"mean_degree": 7}, {"hk_triad": 1.5}, {"ws_rewire": -0.1}, {"beta": 1.5}],
)
def test_invalid_network_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_placement_maps_the_graph_onto_agents():
    graph = graph_for()
    neighbors = place_agents(graph, np.random.default_rng(1))
    # Symmetric, no self-links, and the same degree sequence as the graph.
    for agent, around in enumerate(neighbors):
        assert agent not in around
        assert all(agent in neighbors[other] for other in around)
    assert sorted(len(n) for n in neighbors) == sorted(d for _, d in graph.degree())
    shuffled = place_agents(graph, np.random.default_rng(2))
    assert shuffled != neighbors


# Spreading


def small_config(**overrides):
    """60 agents, one news clue released at step 30, every agent trading every step."""
    base = amm_config(
        n_agents=60,
        turns_per_step=60,
        mean_degree=4,
        n_steps=60,
        n_clues_total=61,
        n_private=60,
        n_news_seeds=1,
    )
    return replace(base, **overrides)


def first_known_steps(model, clue_id):
    first = {}
    while not model.settled:
        model.step()
        for trader in model.traders:
            if clue_id in trader.known:
                first.setdefault(trader.index, model.steps)
    return first


def test_at_beta_one_a_clue_reaches_everyone_within_the_network_diameter():
    """Each agent learns the clue exactly d steps after release,
    where d is its distance from the seed, so everyone reachable knows it within the
    network's diameter."""
    model = MarketModel(small_config(beta=1.0), 0, debug=True)
    first = first_known_steps(model, 60)
    release = model.release_step[60]
    [seed] = [agent for agent, step in first.items() if step == release]
    agents = nx.Graph()
    agents.add_nodes_from(range(60))
    agents.add_edges_from((a, b) for a, around in enumerate(model.neighbors) for b in around)
    distance = nx.single_source_shortest_path_length(agents, seed)
    assert set(first) == set(distance)  # exactly the seed's component
    assert model.learned_step[60] == first
    for agent, d in distance.items():
        assert first[agent] - release == d
    component = agents.subgraph(distance)
    assert max(first.values()) - release <= nx.diameter(component)


def test_at_beta_zero_news_stays_with_its_seeds():
    model = MarketModel(small_config(beta=0.0, n_news_seeds=3), 0)
    model.run()
    assert model.knowers[60] == 3


def test_agents_relay_only_after_their_next_turn():
    model = MarketModel(small_config(beta=1.0), 0)
    trader = model.traders[0]
    trader.learn_heard(60, True)
    assert 60 in trader.pending and not trader.spreading
    model._spread(1)
    assert all(60 not in model.traders[n].known for n in model.neighbors[0])
    trader.finish_turn()
    model._spread(2)
    assert all(60 in model.traders[n].known for n in model.neighbors[0])


def test_spreader_stops_after_telling_someone_who_already_knew():
    model = MarketModel(small_config(beta=1.0), 0)
    trader = model.traders[0]
    trader.learn_heard(60, True)
    trader.finish_turn()
    model._spread(1)  # every neighbor was new: keep spreading
    assert 60 in trader.spreading
    model._spread(2)  # every neighbor now knows: stifled
    assert 60 not in trader.spreading


def test_a_clue_heard_from_many_neighbors_counts_once():
    model = MarketModel(small_config(), 0)
    trader = model.traders[0]
    assert trader.learn_heard(60, True)
    assert not trader.learn_heard(60, True)
    assert not trader.learn_own(60, True, spreadable=True)
    assert (trader.n_heard, trader.heard_net) == (1, 1)


def test_private_clues_never_spread():
    model = MarketModel(amm_config(n_steps=300, n_clues_total=62, beta=1.0), 2)
    model.run()
    dealt = model.traits.n_clues
    # Each agent holds its dealt private clues plus at most the two news clues.
    assert all(t.n_own <= min(dealt[t.index], 60) + 2 for t in model.traders)
    assert all(c >= 60 for t in model.traders for c in t.spreading | t.pending)


def test_half_reach_step_is_recorded():
    model = MarketModel(amm_config(n_steps=400, n_clues_total=62), 1)
    model.run()
    for clue_id in (60, 61):
        assert model.release_step[clue_id] <= model.half_reach_step[clue_id] <= 400


# Disinformation


def test_ratio_sets_the_number_of_fakes():
    ratios = (0, 0.05, 0.1, 0.2, 0.5)
    assert [Config(disinfo_ratio=r).n_fakes for r in ratios] == [0, 1, 2, 4, 10]


def test_fakes_point_against_the_truth_and_are_kept_out_of_p_star():
    model = MarketModel(Config(disinfo_ratio=0.2), 3, debug=True)
    model.run()
    assert model.fake_ids == [80, 81, 82, 83]
    for clue_id in model.fake_ids:
        assert model.points_yes(clue_id) is (not model.world.y)
        assert model.knowers[clue_id] > model.config.n_news_seeds
        assert clue_id not in model._known_real
    known = set().union(*(t.known for t in model.traders)) - set(model.fake_ids)
    assert model.p_star == pytest.approx(model.world.p_star(known))


def test_disinformation_leaves_real_news_and_p_star_unchanged_on_the_same_seed():
    clean = MarketModel(Config(n_steps=600, n_clues_total=66), 4)
    dirty = MarketModel(Config(n_steps=600, n_clues_total=66, disinfo_ratio=0.5), 4)
    clean.run()
    dirty.run()
    assert dirty.config.n_fakes == 3
    real_news = range(60, 66)
    assert [clean.knowers[c] for c in real_news] == [dirty.knowers[c] for c in real_news]
    a = clean.datacollector.get_model_vars_dataframe().p_star
    b = dirty.datacollector.get_model_vars_dataframe().p_star
    assert a.equals(b)


def test_network_settings_do_not_change_the_population_or_world():
    a = MarketModel(Config(hk_triad=0.0), 5)
    b = MarketModel(Config(mean_degree=4), 5)
    assert a.world.y == b.world.y
    assert np.array_equal(a.traits.cash, b.traits.cash)
    assert [t.known for t in a.traders] == [t.known for t in b.traders]
