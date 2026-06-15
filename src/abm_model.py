"""
Greenium Resilience Simulator - Agent-Based Model Core
Author: Chengcheng Gan
Date: 2026-06-12
"""

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import networkx as nx


class InvestorAgent(Agent):
    """
    Institutional investor with heterogeneous beliefs about ESG assets.

    Agent types:
    - passive_index: Tracks ESG indices, low active trading
    - active_esg: Active ESG fund, high greenium sensitivity
    - speculative: Arbitrage-focused, short-term horizon
    - pension: Long-term, regulatory-constrained
    """

    def __init__(self, unique_id, model, agent_type, initial_belief):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.belief_greenwashing = initial_belief  # Prior: P(GW)
        self.learning_rate = self._set_learning_rate()
        self.risk_aversion = self._set_risk_aversion()
        self.portfolio_esg = 0.5  # Initial ESG allocation
        self.wealth = 100.0

    def _set_learning_rate(self):
        """Bayesian learning rate varies by agent type."""
        rates = {
            'passive_index': 0.05,
            'active_esg': 0.15,
            'speculative': 0.25,
            'pension': 0.08
        }
        return rates.get(self.agent_type, 0.10)

    def _set_risk_aversion(self):
        """Risk aversion varies by agent type."""
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

        P(GW|Shock) = P(Shock|GW) * P(GW) / P(Shock)

        regulatory_precision: 0.3 (voluntary) to 0.9 (mandatory SFDR)
        """
        # Likelihood: P(Shock|GW) — higher if greenwashing exists
        likelihood_gw = 0.7 + 0.2 * shock_signal

        # Likelihood: P(Shock|not GW) — baseline climate risk
        likelihood_not_gw = 0.1 + 0.1 * shock_signal

        # Prior
        prior_gw = self.belief_greenwashing
        prior_not_gw = 1 - prior_gw

        # Precision adjustment based on regulatory regime
        precision = regulatory_precision

        # Posterior
        numerator = likelihood_gw * prior_gw * precision
        denominator = numerator + likelihood_not_gw * prior_not_gw * (1 - precision)

        if denominator > 0:
            posterior = numerator / denominator
        else:
            posterior = prior_gw

        # Smooth learning
        self.belief_greenwashing = (
            self.belief_greenwashing * (1 - self.learning_rate) + 
            posterior * self.learning_rate
        )

        # Bound beliefs
        self.belief_greenwashing = np.clip(self.belief_greenwashing, 0.01, 0.99)

    def adjust_portfolio(self, greenium_spread):
        """
        Adjust ESG portfolio allocation based on belief and greenium.

        greenium_spread: current yield discount for ESG assets (bps)
        """
        # Higher belief in greenwashing -> reduce ESG allocation
        esg_penalty = self.belief_greenwashing * 0.5

        # Greenium attractiveness
        greenium_attractiveness = max(0, greenium_spread / 100)

        # Risk-adjusted target allocation
        target_esg = 0.5 - esg_penalty + greenium_attractiveness * 0.3
        target_esg = np.clip(target_esg, 0.0, 1.0)

        # Gradual adjustment
        adjustment_speed = 0.1 / self.risk_aversion
        self.portfolio_esg += (target_esg - self.portfolio_esg) * adjustment_speed
        self.portfolio_esg = np.clip(self.portfolio_esg, 0.0, 1.0)

    def step(self):
        """Execute one trading step."""
        # Get current market state
        shock_signal = self.model.current_shock_signal
        regulatory_precision = self.model.regulatory_precision
        greenium_spread = self.model.current_greenium

        # Update beliefs
        self.update_belief(shock_signal, regulatory_precision)

        # Adjust portfolio
        self.adjust_portfolio(greenium_spread)

        # Trading impact on market
        self._execute_trade()

    def _execute_trade(self):
        """Simplified trade execution affecting market price."""
        # Aggregate demand impact
        trade_size = abs(self.portfolio_esg - 0.5) * self.wealth * 0.01
        self.model.aggregate_demand += trade_size * np.sign(self.portfolio_esg - 0.5)


class ESGMarketModel(Model):
    """
    ESG Market Model with climate shock propagation.
    """

    def __init__(self, n_agents=100, regulatory_regime='mandatory', 
                 network_density=0.15, seed=42):
        # FIX: Mesa 2.3.0 compatibility — call super without seed arg, set manually
        super().__init__()
        self._seed = seed
        np.random.seed(seed)

        self.n_agents = n_agents
        self.regulatory_regime = regulatory_regime
        self.network_density = network_density

        # Regulatory precision mapping
        self.regulatory_precision_map = {
            'baseline': 0.1,
            'voluntary': 0.3,
            'mandatory_l1': 0.6,  # SFDR Article 8
            'mandatory': 0.9,     # SFDR Article 9
            'counter_cyclical': 0.9
        }
        self.regulatory_precision = self.regulatory_precision_map.get(
            regulatory_regime, 0.5
        )

        # Market state
        self.current_shock_signal = 0.0
        self.current_greenium = 15.0  # Initial: 15 bps
        self.aggregate_demand = 0.0
        self.shock_history = []
        self.greenium_history = []

        # Network
        self.network = self._build_network()
        self.grid = NetworkGrid(self.network)

        # Schedule
        self.schedule = RandomActivation(self)

        # Create agents
        self._create_agents()

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                'Greenium': lambda m: m.current_greenium,
                'Shock_Signal': lambda m: m.current_shock_signal,
                'Avg_Belief_GW': lambda m: np.mean([
                    a.belief_greenwashing for a in m.schedule.agents
                ]),
                'ESG_Demand': lambda m: m.aggregate_demand,
                'Network_Centrality_Max': lambda m: max(
                    nx.eigenvector_centrality(m.network).values()
                ) if m.network else 0
            },
            agent_reporters={
                'Belief_GW': lambda a: a.belief_greenwashing,
                'Portfolio_ESG': lambda a: a.portfolio_esg,
                'Agent_Type': lambda a: a.agent_type
            }
        )

    def _build_network(self):
        """Build random network with specified density."""
        n = self.n_agents
        m = int(n * self.network_density / 2)
        # FIX: Ensure seed is integer, avoid float random state errors
        G = nx.barabasi_albert_graph(n, max(m, 1), seed=self._seed)
        return G

    def _create_agents(self):
        """Create heterogeneous agent population."""
        agent_types = ['passive_index', 'active_esg', 'speculative', 'pension']
        type_weights = [0.3, 0.3, 0.2, 0.2]

        for i in range(self.n_agents):
            agent_type = np.random.choice(agent_types, p=type_weights)
            initial_belief = np.random.beta(2, 5)  # Skewed low: most believe low GW

            agent = InvestorAgent(i, self, agent_type, initial_belief)
            self.schedule.add(agent)

            # Place on network
            self.grid.place_agent(agent, i)

    def _generate_climate_shock(self, step):
        """Generate climate shock signal."""
        # Base probability: 5% per step
        if np.random.random() < 0.05:
            # Shock severity: 0.3 to 1.0
            severity = np.random.uniform(0.3, 1.0)

            # Affected nodes: 10-30% of network
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
        # Base decay: greenium mean-reverts to 10 bps
        base_greenium = 10.0

        # Demand impact: positive demand -> greenium widens
        demand_impact = self.aggregate_demand * 0.5

        # Shock impact: recent shocks compress greenium
        shock_impact = 0.0
        if self.shock_history:
            recent_shocks = [s for s in self.shock_history 
                             if s['step'] > self.schedule.steps - 10]
            if recent_shocks:
                avg_severity = np.mean([s['severity'] for s in recent_shocks])
                shock_impact = -avg_severity * 20  # Max 20 bps compression

        # Network centrality amplification
        centrality = nx.eigenvector_centrality(self.network)
        max_centrality = max(centrality.values())
        centrality_effect = 0.0
        if max_centrality > 0.23:  # Threshold from manuscript
            centrality_effect = (max_centrality - 0.23) * 10

        # Update
        target = base_greenium + demand_impact + shock_impact + centrality_effect
        self.current_greenium += (target - self.current_greenium) * 0.1
        self.current_greenium = max(0, self.current_greenium)

        # Reset aggregate demand
        self.aggregate_demand = 0.0

    def step(self):
        """Execute one model step."""
        # Generate climate shock
        self.current_shock_signal = self._generate_climate_shock(self.schedule.steps)

        # Activate all agents
        self.schedule.step()

        # Update market state
        self._update_greenium()
        self.greenium_history.append(self.current_greenium)

        # Collect data
        self.datacollector.collect(self)

    def run_simulation(self, n_steps=100):
        """Run full simulation."""
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()


def run_monte_carlo(n_runs=100, n_steps=100, regulatory_regime='mandatory'):
    """
    Run Monte Carlo simulation across multiple seeds.

    Returns summary statistics for greenium dynamics.
    """
    results = []

    for run in range(n_runs):
        # FIX: Use different seed per run for true Monte Carlo
        model = ESGMarketModel(
            n_agents=100,
            regulatory_regime=regulatory_regime,
            network_density=0.15,
            seed=run + 42
        )
        df = model.run_simulation(n_steps)

        # Extract key metrics
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
    # Quick test
    print("Running baseline simulation...")
    model = ESGMarketModel(n_agents=50, regulatory_regime='mandatory')
    results = model.run_simulation(n_steps=50)
    print(f"Final greenium: {results['Greenium'].iloc[-1]:.2f} bps")
    print(f"Average belief in GW: {results['Avg_Belief_GW'].iloc[-1]:.3f}")
    print(f"Number of shocks: {len(model.shock_history)}")
    print("Simulation complete.")