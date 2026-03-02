# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is an AI/ML study monorepo ("AIStudy") managed as a **uv workspace** with three sub-packages under `packages/`:

- `reinforcement_learning` — RL algorithms (PPO, DQN, SARSA, policy gradient, etc.) with Jupyter notebooks and scripts
- `artificial_neural_network` — PyTorch / PyTorch Lightning demos (MNIST CNN)
- `open_ai_api_test` — OpenAI-compatible API tests (requires external LLM endpoint, e.g. Ollama)

### Running scripts

- Run standalone scripts: `uv run python packages/<package>/src/<module>/<script>.py`
- Run workspace package entry points: `uv run --package <package-name> python -c "from <module> import main; main()"`
- Launch Jupyter for notebooks: `uv run jupyter notebook --no-browser`
- Set `MPLBACKEND=Agg` when running scripts with matplotlib in headless environments to avoid display errors.

### Build / compile gotchas

- The default `c++` on this VM is clang, which fails to find `<string>` and other C++ standard library headers. Set `CC=gcc CXX=g++` before running `uv sync` to ensure `box2d-py` (required by `gymnasium[all]`) compiles correctly.
- System dependencies required: `swig`, `g++` (via `build-essential`), `libstdc++-13-dev`.
- Root `pyproject.toml` requires Python ≥3.13. The sub-package `.python-version` files say 3.12 but the workspace-level constraint takes precedence. Use `uv python install 3.13` if needed.

### No linting or tests

This repository has no linting configuration (ruff, mypy, flake8, etc.) and no automated test suite. Verification is done by running the scripts and notebooks directly.
