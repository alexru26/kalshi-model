"""The social network and how news and disinformation spread on it (PLAN.md Section 2.4).

The network is a Holme-Kim graph (NetworkX `powerlaw_cluster_graph(N, k / 2, p)`):
preferential attachment gives it hubs, and closing a triangle with probability
`hk_triad` after each preferential link gives it clustering.

Spreading rules: private clues never spread; only news and fake clues do. An agent that
learns one (as a seed or from a neighbor) keeps it until it has had its next turn to
trade (van Bommel 2003), then becomes a spreader. Each step a spreader tells each
neighbor with probability `beta`. Like Daley & Kendall's (1964) stiflers, a spreader
stops retelling a clue once it has told it to someone who already knew it. A told
neighbor counts a true news clue as its own evidence (it can be checked at its source)
and a fake as heard; whoever already knows a clue ignores it, so each clue counts once
however many neighbors relay it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from kalshi_model.config import Config


def build_network(config: Config, rng: np.random.Generator) -> nx.Graph:
    """The Holme-Kim network on nodes 0..N-1, drawn from `rng` (the "network" stream)."""
    seed = int(rng.integers(2**32))
    if config.mean_degree == 0:
        return nx.empty_graph(config.n_agents)
    graph = nx.powerlaw_cluster_graph(
        config.n_agents, config.mean_degree // 2, config.hk_triad, seed=seed
    )
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
