# Greenium Resilience Simulator

Agent-based model for ESG market greenium dynamics under climate shocks.

## Environment

- Python 3.10+
- Dependencies: see `requirements.txt`

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m tests.test_abm

# Run main simulation
python -m src.abm_model

# Run regulatory sandbox
python -m src.regulatory_sandbox