"""Social networks and how news and disinformation spread on them (PLAN.md Section 2.4).

Every network has `n_agents` nodes and the same mean degree, so only its shape differs:

| Name          | NetworkX generator                    | Adds                         |
|---------------|---------------------------------------|------------------------------|
| erdos_renyi   | gnp_random_graph(N, k / (N - 1))      | nothing (null baseline)      |
| watts_strogatz| connected_watts_strogatz_graph(N, k)  | clustering, a few long links |
| barabasi_albert| barabasi_albert_graph(N, k / 2)      | hubs                         |
| holme_kim     | powerlaw_cluster_graph(N, k / 2, p)   | hubs and clustering          |
| sbm           | stochastic_block_model, two halves    | two communities              |

Spreading rules: private clues never spread; only news and fake clues do. An agent that
learns one (as a seed or from a neighbor) keeps it until it has had its next turn to
trade (van Bommel 2003), then becomes a spreader. Each step a spreader tells each
neighbor with probability `beta`. Like Daley & Kendall's (1964) stiflers, a spreader
stops retelling a clue once it has told it to someone who already knew it. A told
neighbor counts the clue as heard; whoever already knows a clue ignores it, so each
clue counts once however many neighbors relay it.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np

from kalshi_model.config import Config


def _erdos_renyi(config: Config, seed: int) -> nx.Graph:
    n = config.n_agents
    return nx.gnp_random_graph(n, config.mean_degree / (n - 1), seed=seed)


def _watts_strogatz(config: Config, seed: int) -> nx.Graph:
    return nx.connected_watts_strogatz_graph(
        config.n_agents, config.mean_degree, config.ws_rewire, seed=seed
    )


def _barabasi_albert(config: Config, seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(config.n_agents, config.mean_degree // 2, seed=seed)


def _holme_kim(config: Config, seed: int) -> nx.Graph:
    return nx.powerlaw_cluster_graph(
        config.n_agents, config.mean_degree // 2, config.hk_triad, seed=seed
    )


def _sbm(config: Config, seed: int) -> nx.Graph:
    # Two equal groups. A share `sbm_cross_share` of each node's expected links cross over.
    n = config.n_agents
    sizes = [n // 2, n - n // 2]
    k, cross = config.mean_degree, config.sbm_cross_share
    p_in = k * (1 - cross) / (sizes[0] - 1)
    p_out = k * cross / sizes[1]
    return nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]], seed=seed)


GENERATORS: dict[str, Callable[[Config, int], nx.Graph]] = {
    "erdos_renyi": _erdos_renyi,
    "watts_strogatz": _watts_strogatz,
    "barabasi_albert": _barabasi_albert,
    "holme_kim": _holme_kim,
    "sbm": _sbm,
}


def build_network(config: Config, rng: np.random.Generator) -> nx.Graph:
    """The configured network on nodes 0..N-1, drawn from `rng` (the "network" stream)."""
    seed = int(rng.integers(2**32))
    if config.mean_degree == 0:
        return nx.empty_graph(config.n_agents)
    graph = GENERATORS[config.network](config, seed)
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def place_agents(graph: nx.Graph, rng: np.random.Generator) -> list[list[int]]:
    """Put agents on nodes at random; return each agent's neighbors as agent indices."""
    n = graph.number_of_nodes()
    agent_at = rng.permutation(n)  # node -> agent
    node_of = np.empty(n, dtype=np.int64)
    node_of[agent_at] = np.arange(n)
    return [sorted(int(agent_at[v]) for v in graph.neighbors(int(node_of[a]))) for a in range(n)]
