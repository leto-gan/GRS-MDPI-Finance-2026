"""
Temporal Network Builder for GRS
Author: Chengcheng Gan
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


class TemporalNetwork:
    """
    Multi-layer temporal network for ESG asset contagion.

    Layers:
    - Sector co-movement (correlation-based)
    - Shared ESG rating providers
    - Cross-ownership (institutional holdings)
    """

    def __init__(self):
        self.network = nx.Graph()
        self.layers = {}
        self.node_attributes = {}

    def build_from_sample_data(self, n_firms=50, seed=None):
        """
        Build sample network for demonstration.

        In production, use SEC 13F filings and ownership data.
        """
        np.random.seed(seed)

        # Create firms with attributes
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

        # Layer 1: Sector co-movement (high within-sector connectivity)
        for i, firm1 in enumerate(firms):
            for j, firm2 in enumerate(firms[i+1:], i+1):
                if self.node_attributes[firm1]['sector'] == self.node_attributes[firm2]['sector']:
                    if np.random.random() < 0.3:  # 30% within-sector connection
                        self.network.add_edge(firm1, firm2, 
                                            layer='sector', 
                                            weight=np.random.uniform(0.5, 1.0))

        # Layer 2: Shared ESG rating provider
        for i, firm1 in enumerate(firms):
            for j, firm2 in enumerate(firms[i+1:], i+1):
                if self.node_attributes[firm1]['esg_provider'] == self.node_attributes[firm2]['esg_provider']:
                    if np.random.random() < 0.2:  # 20% provider connection
                        self.network.add_edge(firm1, firm2,
                                            layer='esg_provider',
                                            weight=np.random.uniform(0.3, 0.8))

        # Layer 3: Cross-ownership (random institutional holdings)
        n_institutions = 10
        for inst in range(n_institutions):
            holdings = np.random.choice(firms, size=np.random.randint(3, 15), replace=False)
            for i, firm1 in enumerate(holdings):
                for firm2 in holdings[i+1:]:
                    if not self.network.has_edge(firm1, firm2):
                        self.network.add_edge(firm1, firm2,
                                            layer='ownership',
                                            weight=np.random.uniform(0.1, 0.5))

        return self.network

    def compute_eigenvector_centrality(self):
        """Compute eigenvector centrality for all nodes."""
        if not self.network:
            return {}
        try:
            centrality = nx.eigenvector_centrality(self.network, weight='weight')
            return centrality
        except:
            # Fallback to degree centrality if eigenvector fails
            return dict(self.network.degree(weight='weight'))

    def simulate_shock_propagation(self, seed_nodes, shock_severity=0.5, 
                                   decay_factor=0.8, max_hops=3):
        """
        Simulate shock propagation through network.

        Returns: dict of node -> shock intensity
        """
        shock_intensity = {node: 0.0 for node in self.network.nodes()}

        # Initialize seed nodes
        for node in seed_nodes:
            if node in shock_intensity:
                shock_intensity[node] = shock_severity

        # Propagate through network
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
            edges = [(u, v) for u, v, d in self.network.edges(data=True) 
                     if d.get('layer') == layer]
            stats[layer] = {
                'n_edges': len(edges),
                'avg_weight': np.mean([self.network[u][v]['weight'] for u, v in edges]) if edges else 0
            }
        return stats


if __name__ == '__main__':
    print("Building sample network...")
    network = TemporalNetwork()
    G = network.build_from_sample_data(n_firms=50, seed=42)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    centrality = network.compute_eigenvector_centrality()
    print(f"Max centrality: {max(centrality.values()):.4f}")

    stats = network.get_layer_statistics()
    for layer, info in stats.items():
        print(f"{layer}: {info['n_edges']} edges, avg weight {info['avg_weight']:.3f}")

    # Simulate shock
    shock = network.simulate_shock_propagation(['FIRM_001'], shock_severity=0.7)
    affected = sum(1 for v in shock.values() if v > 0.1)
    print(f"Shock affected {affected} firms")
