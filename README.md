# ARC-AGI — Seamless FunctionGemma & Toroidal MoM Prototype

Project goal
- Build a reproducible codebase for the ARC-AGI contest, integrating HRM-style specialists, FunctionGemma as a gated micro-agent, and a toroidal / "Seamless" layer topology applied selectively to improve sample efficiency and determinism.

Contents
- scripts/: utilities and patch scripts (Seamless FunctionGemma wrapper).
- notebooks/: Colab-ready notebooks for interactive experiments.
- src/: model wrappers, trainers, and evaluation harness.
- ci/: GitHub Actions workflows for linting and smoke tests.

Quickstart
- Clone the repo
- Create a Python venv and install requirements:
  pip install -r requirements.txt
- To patch a local FunctionGemma checkout, run:
  python scripts/seamless_functiongemma_patch.py

Notes
- This repo intentionally does not include large model checkpoints. Provide HF tokens / local checkpoints as needed for experiments.
- See docs/ for architecture rationale, experiment logs, and runbooks.

License
- Default: MIT (see LICENSE file)
