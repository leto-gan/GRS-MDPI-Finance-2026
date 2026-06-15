#!/usr/bin/env python3
"""
Greenium Resilience Simulator - Agent-Based Model Core
Author: Chengcheng Gan
Date: 2026-06-15

Mesa 2.3.0 compatible. Multi-layer network integration (487 nodes).
"""

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx

# FIX: Import multi-layer network builder
from network_builder import TemporalNetwork


class InvestorAgent(Agent):
    """
    Institutional investor with heterogeneous beliefs about ESG assets.
    """

    def __init__(self, unique_id, model, agent_type, initial_belief):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.belief_greenwashing = initial_belief
        self.learning_rate = self._set_learning_rate()
        self.risk_aversion = self._set_risk_aversion()
        self.portfolio_esg = 0.5
        self.wealth = 100.0

    def _set_learning_rate(self):
        rates = {
            'passive_index': 0.05,
            'active_esg': 0.15,
            'speculative': 0.25,
            'pension': 0.08
        }
        return rates.get(self.agent_type, 0.10)

    def _set_risk_aversion(self):
        aversion = {
            'passive_index': 2.0,
            'active_esg': 1.5,
            'speculative': 0.8,
            'pension': 3.0
        }
        return aversion.get(self.agent_type, 1.5)

    def update_belief(self, shock_signal, regulatory_precision):
        """
        Bayesian belief updating about greenwashing.
        """
        likelihood_gw = 0.7 + 0.2 * shock_signal
        likelihood_not_gw = 0.1 + 0.1 * shock_signal
        prior_gw = self.belief_greenwashing
        prior_not_gw = 1 - prior_gw

        precision = np.clip(regulatory_precision, 0.01, 0.99)
        lr = (likelihood_gw / max(likelihood_not_gw, 1e-9)) * (precision / (1 - precision))

        posterior = lr * prior_gw / (1 + lr * prior_gw - prior_gw)
        posterior = np.clip(posterior, 0.01, 0.99)

        self.belief_greenwashing = (
            self.belief_greenwashing * (1 - self.learning_rate) +
            posterior * self.learning_rate
        )
        self.belief_greenwashing = np.clip(self.belief_greenwashing, 0.01, 0.99)

    def adjust_portfolio(self, greenium_spread):
        """
        Adjust ESG portfolio allocation based on belief and greenium.
        """
        esg_penalty = self.belief_greenwashing * 0.5
        greenium_attractiveness = max(0, greenium_spread / 100)
        target_esg = 0.5 - esg_penalty + greenium_attractiveness * 0.3
        target_esg = np.clip(target_esg, 0.0, 1.0)

        adjustment_speed = 0.1 / self.risk_aversion
        self.portfolio_esg += (target_esg - self.portfolio_esg) * adjustment_speed
        self.portfolio_esg = np.clip(self.portfolio_esg, 0.0, 1.0)

    def step(self):
        """Execute one trading step."""
        shock_signal = self.model.current_shock_signal
        regulatory_precision = self.model.regulatory_precision
        greenium_spread = self.model.current_greenium

        self.update_belief(shock_signal, regulatory_precision)
        self.adjust_portfolio(greenium_spread)
        self._execute_trade()

    def _execute_trade(self):
        """Simplified trade execution affecting market price."""
        trade_size = abs(self.portfolio_esg - 0.5) * self.wealth * 0.01
        self.model.aggregate_demand += trade_size * np.sign(self.portfolio_esg - 0.5)


class ESGMarketModel(Model):
    """
    ESG Market Model with climate shock propagation.
    Mesa 2.3.0 compatible. Multi-layer network.
    """

    def __init__(self, n_agents=487, regulatory_regime='mandatory',
                 network_density=0.15, seed=42):
        super().__init__(seed=seed)
        self._seed = seed
        np.random.seed(seed)

        self.n_agents = n_agents
        self.regulatory_regime = regulatory_regime
        self.network_density = network_density

        self.regulatory_precision_map = {
            'baseline': 0.1,
            'voluntary': 0.3,
            'mandatory_l1': 0.6,
            'mandatory': 0.9,
            'counter_cyclical': 0.9
        }
        self.regulatory_precision = self.regulatory_precision_map.get(
            regulatory_regime, 0.5
        )

        self.current_shock_signal = 0.0
        self.current_greenium = 15.0
        self.aggregate_demand = 0.0
        self.shock_history = []
        self.greenium_history = []

        # FIX: Multi-layer temporal network (sector + ESG provider + ownership)
        self.network = self._build_network()

        self.schedule = RandomActivation(self)
        self._create_agents()

        # Centrality cached once (static network)
        self._cached_centrality = self._compute_cached_centrality()

        self.datacollector = DataCollector(
            model_reporters={
                'Greenium': lambda m: m.current_greenium,
                'Shock_Signal': lambda m: m.current_shock_signal,
                'Avg_Belief_GW': lambda m: np.mean([
                    a.belief_greenwashing for a in m.schedule.agents
                ]),
                'ESG_Demand': lambda m: m.aggregate_demand,
                'Network_Centrality_Max': lambda m: max(
                    self._cached_centrality.values()
                ) if self._cached_centrality else 0
            },
            agent_reporters={
                'Belief_GW': lambda a: a.belief_greenwashing,
                'Portfolio_ESG': lambda a: a.portfolio_esg,
                'Agent_Type': lambda a: a.agent_type
            }
        )

    def _build_network(self):
        """
        Build multi-layer temporal network via TemporalNetwork.
        Layers: sector co-movement, shared ESG rating providers, cross-ownership.
        Replaces single-layer Barabási-Albert.
        """
        builder = TemporalNetwork()
        # Returns integer-node graph (0..n-1), compatible with Mesa Agent unique_id
        G = builder.build_mesa_compatible_graph(n_firms=self.n_agents, seed=self._seed)
        return G

    def _compute_cached_centrality(self):
        """Compute eigenvector centrality once (static network)."""
        try:
            return nx.eigenvector_centrality(self.network, weight='weight')
        except Exception:
            return dict(self.network.degree())

    def _create_agents(self):
        """Create heterogeneous agent population."""
        agent_types = ['passive_index', 'active_esg', 'speculative', 'pension']
        type_weights = [0.3, 0.3, 0.2, 0.2]

        for i in range(self.n_agents):
            agent_type = np.random.choice(agent_types, p=type_weights)
            initial_belief = np.random.beta(2, 5)
            agent = InvestorAgent(i, self, agent_type, initial_belief)
            self.schedule.add(agent)

    def _generate_climate_shock(self, step):
        """Generate climate shock signal."""
        if np.random.random() < 0.05:
            severity = np.random.uniform(0.3, 1.0)
            affected_fraction = np.random.uniform(0.1, 0.3)
            n_affected = int(self.n_agents * affected_fraction)
            affected_nodes = np.random.choice(
                self.n_agents, n_affected, replace=False
            )
            shock = {
                'severity': severity,
                'affected_nodes': affected_nodes,
                'step': step
            }
            self.shock_history.append(shock)
            return severity
        return 0.0

    def _update_greenium(self):
        """Update greenium based on aggregate demand and shocks."""
        base_greenium = 10.0
        demand_impact = self.aggregate_demand * 0.5

        shock_impact = 0.0
        if self.shock_history:
            recent_shocks = [
                s for s in self.shock_history
                if s['step'] > self.schedule.steps - 10
            ]
            if recent_shocks:
                avg_severity = np.mean([s['severity'] for s in recent_shocks])
                shock_impact = -avg_severity * 20

        centrality_effect = 0.0
        if self._cached_centrality:
            max_centrality = max(self._cached_centrality.values())
            if max_centrality > 0.23:
                centrality_effect = (max_centrality - 0.23) * 10

        target = base_greenium + demand_impact + shock_impact + centrality_effect
        self.current_greenium += (target - self.current_greenium) * 0.1
        # FIX: Enforce greenium floor at 0 bps (manuscript definition)
        self.current_greenium = max(0.0, float(self.current_greenium))

        self.aggregate_demand = 0.0

    def step(self):
        """Execute one model step."""
        self.current_shock_signal = self._generate_climate_shock(self.schedule.steps)
        self.schedule.step()
        self._update_greenium()
        self.greenium_history.append(self.current_greenium)
        self.datacollector.collect(self)

    def run_simulation(self, n_steps=100):
        """Run full simulation."""
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()


def run_monte_carlo(n_runs=100, n_steps=100, regulatory_regime='mandatory'):
    """
    Run Monte Carlo simulation across multiple seeds.
    """
    results = []
    for run in range(n_runs):
        model = ESGMarketModel(
            n_agents=487,
            regulatory_regime=regulatory_regime,
            network_density=0.15,
            seed=run + 42
        )
        df = model.run_simulation(n_steps)
        results.append({
            'run': run,
            'final_greenium': df['Greenium'].iloc[-1],
            'min_greenium': df['Greenium'].min(),
            'max_greenium': df['Greenium'].max(),
            'volatility': df['Greenium'].std(),
            'n_shocks': len(model.shock_history),
            'avg_belief_final': df['Avg_Belief_GW'].iloc[-1]
        })
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Running baseline simulation (n=487, multi-layer network)...")
    model = ESGMarketModel(n_agents=487, regulatory_regime='mandatory', seed=42)
    results = model.run_simulation(n_steps=50)
    print(f"Final greenium: {results['Greenium'].iloc[-1]:.2f} bps")
    print(f"Average belief in GW: {results['Avg_Belief_GW'].iloc[-1]:.3f}")
    print(f"Number of shocks: {len(model.shock_history)}")
    print("Simulation complete.")