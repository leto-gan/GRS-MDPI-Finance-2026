"""
Unit tests for ABM model
"""

import sys
sys.path.insert(0, 'src')

import unittest
import numpy as np
from src.abm_model import InvestorAgent, ESGMarketModel


class TestInvestorAgent(unittest.TestCase):

    def setUp(self):
        self.model = ESGMarketModel(n_agents=10, seed=42)
        self.agent = InvestorAgent(0, self.model, 'active_esg', 0.2)

    def test_initial_belief(self):
        self.assertEqual(self.agent.belief_greenwashing, 0.2)

    def test_learning_rate(self):
        self.assertEqual(self.agent.learning_rate, 0.15)

    def test_belief_update(self):
        initial = self.agent.belief_greenwashing
        self.agent.update_belief(shock_signal=0.8, regulatory_precision=0.9)
        self.assertNotEqual(self.agent.belief_greenwashing, initial)
        self.assertGreater(self.agent.belief_greenwashing, initial)  # Should increase

    def test_portfolio_adjustment(self):
        self.agent.portfolio_esg = 0.5
        self.agent.belief_greenwashing = 0.8  # High belief -> reduce ESG
        self.agent.adjust_portfolio(greenium_spread=10)
        self.assertLess(self.agent.portfolio_esg, 0.5)


class TestESGMarketModel(unittest.TestCase):

    def setUp(self):
        self.model = ESGMarketModel(n_agents=20, regulatory_regime='mandatory', seed=42)

    def test_initialization(self):
        self.assertEqual(self.model.n_agents, 20)
        self.assertEqual(self.model.regulatory_regime, 'mandatory')

    def test_network_creation(self):
        self.assertEqual(self.model.network.number_of_nodes(), 20)

    def test_agent_creation(self):
        self.assertEqual(len(self.model.schedule.agents), 20)

    def test_simulation_run(self):
        results = self.model.run_simulation(n_steps=10)
        self.assertEqual(len(results), 10)
        self.assertIn('Greenium', results.columns)


if __name__ == '__main__':
    unittest.main()
