#!/usr/bin/env python3
"""
Temporal Network Builder for GRS
Author: Chengcheng Gan
Date: 2026-06-15

Multi-layer network: sector co-movement, shared ESG rating providers, cross-ownership.
Mesa-compatible integer-node graph output added.
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt


class TemporalNetwork:
    """
    Multi-layer temporal network for ESG asset contagion.
    """

    def __init__(self):
        self.network = nx.Graph()
        self.layers = {}
        self.node_attributes = {}
        self.node_to_id = {}
        self.id_to_node = {}

    def build_from_sample_data(self, n_firms=487, seed=None):
        """
        Build sample network for demonstration.
        In production, use SEC 13F filings and ownership data.
        """
        np.random.seed(seed)

        firms = [f'FIRM_{i:03d}' for i in range(n_firms)]
        sectors = ['Energy', 'Tech', 'Finance', 'Healthcare', 'Consumer']
        esg_providers = ['Sustainalytics', 'MSCI', 'Refinitiv', 'CDP']

        for firm in firms:
            self.node_attributes[firm] = {
                'sector': np.random.choice(sectors),
                'esg_provider': np.random.choice(esg_providers),
                'market_cap': np.random.lognormal(10, 1),
                'esg_score': np.random.normal(60, 15)
            }
            self.network.add_node(firm, **self.node_attributes[firm])

        # Layer 1: Sector co-movement
        for i, firm1 in enumerate(firms):
            for j, firm2 in enumerate(firms[i+1:], i+1):
                if self.node_attributes[firm1]['sector'] == self.node_attributes[firm2]['sector']:
                    if np.random.random() < 0.3:
                        self.network.add_edge(
                            firm1, firm2,
                            layer='sector',
                            weight=np.random.uniform(0.5, 1.0)
                        )

        # Layer 2: Shared ESG rating provider
        for i, firm1 in enumerate(firms):
            for j, firm2 in enumerate(firms[i+1:], i+1):
                if self.node_attributes[firm1]['esg_provider'] == self.node_attributes[firm2]['esg_provider']:
                    if np.random.random() < 0.2:
                        self.network.add_edge(
                            firm1, firm2,
                            layer='esg_provider',
                            weight=np.random.uniform(0.3, 0.8)
                        )

        # Layer 3: Cross-ownership
        n_institutions = 10
        for inst in range(n_institutions):
            holdings = np.random.choice(
                firms, size=np.random.randint(3, 15), replace=False
            )
            for i, firm1 in enumerate(holdings):
                for firm2 in holdings[i+1:]:
                    if not self.network.has_edge(firm1, firm2):
                        self.network.add_edge(
                            firm1, firm2,
                            layer='ownership',
                            weight=np.random.uniform(0.1, 0.5)
                        )

        return self.network

    def build_mesa_compatible_graph(self, n_firms=487, seed=None):
        """
        Build network and return integer-node graph for Mesa ABM compatibility.
        """
        self.build_from_sample_data(n_firms=n_firms, seed=seed)

        node_list = sorted(self.network.nodes())
        self.node_to_id = {node: i for i, node in enumerate(node_list)}
        self.id_to_node = {i: node for i, node in enumerate(node_list)}

        G_int = nx.Graph()
        for i in range(n_firms):
            G_int.add_node(i, **self.node_attributes[self.id_to_node[i]])

        for u, v, data in self.network.edges(data=True):
            G_int.add_edge(
                self.node_to_id[u],
                self.node_to_id[v],
                **data
            )

        return G_int

    def get_adjacency_matrix(self, weighted=True):
        """Return adjacency matrix as numpy array (n_firms × n_firms)."""
        if not self.network:
            return None

        nodes = sorted(self.network.nodes())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        adj = np.zeros((n, n))

        for u, v, data in self.network.edges(data=True):
            weight = data.get('weight', 1.0) if weighted else 1.0
            i, j = node_idx[u], node_idx[v]
            adj[i, j] = weight
            adj[j, i] = weight

        return adj

    def compute_eigenvector_centrality(self):
        """Compute eigenvector centrality for all nodes."""
        if not self.network:
            return {}
        try:
            return nx.eigenvector_centrality(self.network, weight='weight')
        except Exception:
            return dict(self.network.degree(weight='weight'))

    def simulate_shock_propagation(self, seed_nodes, shock_severity=0.5,
                                   decay_factor=0.8, max_hops=3):
        """
        Simulate shock propagation through network.
        """
        shock_intensity = {node: 0.0 for node in self.network.nodes()}

        for node in seed_nodes:
            if node in shock_intensity:
                shock_intensity[node] = shock_severity

        current_frontier = set(seed_nodes)
        for hop in range(max_hops):
            next_frontier = set()
            for node in current_frontier:
                for neighbor in self.network.neighbors(node):
                    if neighbor not in seed_nodes:
                        edge_weight = self.network[node][neighbor].get('weight', 0.5)
                        propagated = shock_intensity[node] * edge_weight * decay_factor
                        if propagated > shock_intensity[neighbor]:
                            shock_intensity[neighbor] = propagated
                            next_frontier.add(neighbor)
            current_frontier = next_frontier
            if not current_frontier:
                break

        return shock_intensity

    def get_layer_statistics(self):
        """Return statistics for each network layer."""
        stats = {}
        for layer in ['sector', 'esg_provider', 'ownership']:
            edges = [
                (u, v) for u, v, d in self.network.edges(data=True)
                if d.get('layer') == layer
            ]
            stats[layer] = {
                'n_edges': len(edges),
                'avg_weight': np.mean([
                    self.network[u][v]['weight'] for u, v in edges
                ]) if edges else 0
            }
        return stats

    def plot_degree_distribution(self, output_path='degree_distribution.png'):
        """Plot degree distribution for manuscript Figure 3 supplement."""
        degrees = [d for n, d in self.network.degree()]
        plt.figure(figsize=(8, 5))
        plt.hist(degrees, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution (Multi-Layer ESG Network, n=487)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Degree distribution saved to {output_path}")


if __name__ == '__main__':
    print("Building sample network (n=487)...")
    network = TemporalNetwork()
    G = network.build_from_sample_data(n_firms=487, seed=42)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    centrality = network.compute_eigenvector_centrality()
    print(f"Max centrality: {max(centrality.values()):.4f}")

    stats = network.get_layer_statistics()
    for layer, info in stats.items():
        print(f"{layer}: {info['n_edges']} edges, avg weight {info['avg_weight']:.3f}")

    # Test Mesa-compatible graph
    G_int = network.build_mesa_compatible_graph(n_firms=487, seed=42)
    print(f"Mesa-compatible graph: {G_int.number_of_nodes()} nodes, {G_int.number_of_edges()} edges")

    # Test adjacency matrix
    adj = network.get_adjacency_matrix(weighted=True)
    print(f"Adjacency matrix shape: {adj.shape}")

    # Generate degree distribution plot
    network.plot_degree_distribution('degree_distribution.png')

    # Simulate shock
    shock = network.simulate_shock_propagation(['FIRM_001'], shock_severity=0.7)
    affected = sum(1 for v in shock.values() if v > 0.1)
    print(f"Shock affected {affected} firms")