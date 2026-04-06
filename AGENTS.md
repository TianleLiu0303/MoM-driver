# AGENTS.md

This file guides agentic coding tools working in this repository.
Keep commands and conventions aligned with NAVSIM’s existing workflows.

## Repository Orientation

- Primary package: `navsim/`
- Training scripts: `navsim/planning/script/`
- Agent implementations: `navsim/agents/`
- Datasets and caching: `navsim/planning/metric_caching/`, `navsim/planning/training/`
- Helper scripts: `scripts/` (training, evaluation, submission)
- Docs: `docs/`

## Environment Setup (local)

- Create conda env:
  - `conda env create --name navsim -f environment.yml`
  - `conda activate navsim`
  - `pip install -e .`
- Required env vars (see `docs/install.md`):
  - `NUPLAN_MAP_VERSION`, `NUPLAN_MAPS_ROOT`, `NAVSIM_EXP_ROOT`,
    `NAVSIM_DEVKIT_ROOT`, `OPENSCENE_DATA_ROOT`
- Data download scripts: `download/` (maps + splits)

## Build, Lint, Test

Build is an editable install (no compiled build step in repo).

- Install in editable mode: `pip install -e .`
- Lint (flake8 config present in `.flake8`):
  - `flake8 navsim`
  - Configuration: max line length 120, several ignores
- Tests:
  - No `tests/` directory or `pytest.ini` in repo.
  - If tests are added, use: `pytest path/to/test_file.py::test_name`
  - Pytest is listed in `requirements.txt`

## Common Run Commands

Use the scripts in `scripts/` as the reference for training/eval/caching.

- Training entrypoint:
  - `python navsim/planning/script/run_training.py agent=transfuser_agent`
  - See `scripts/training/run_transfuser_training.sh` for full env vars
- Metric caching:
  - `python navsim/planning/script/run_metric_caching.py train_test_split=navtest`
  - See `scripts/evaluation/run_metric_caching.sh`
- PDM score evaluation:
  - `python navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=transfuser_agent`
  - See `scripts/evaluation/run_transfuser.sh` for args
- Submission helpers:
  - `navsim/planning/script/run_create_submission_pickle.py`
  - `navsim/planning/script/run_merge_submission_pickles.py`

## Code Style Guidelines

### Formatting

- Indent with 4 spaces; no tabs.
- Line length: 120 (per `.flake8`).
- Use blank lines to separate logical sections.
- Prefer explicit parentheses for long argument lists (as in scripts).

### Imports

- Group imports in this order:
  1. Standard library
  2. Third-party packages
  3. Local `navsim` imports
- Keep one import per line where possible.
- Avoid wildcard imports.

### Naming

- Modules and functions: `snake_case`.
- Classes: `CamelCase`.
- Constants: `UPPER_SNAKE_CASE` (e.g., `CONFIG_PATH`).
- Boolean flags use `is_`/`has_`/`use_` prefixes.

### Typing

- Use type hints for function signatures and key variables.
- Use `Dict`, `List`, `Tuple`, `Union` from `typing` (consistent with existing code).
- For tensors, annotate as `torch.Tensor` where relevant.

### Docstrings

- Use triple-quoted docstrings with `:param` / `:return` style.
- Describe inputs and outputs clearly, especially for data builders and agents.

### Error Handling & Logging

- Prefer `logging.getLogger(__name__)` and structured log messages.
- Use `assert` for configuration invariants in scripts.
- Raise `NotImplementedError` for unimplemented interfaces (see `AbstractAgent`).

### Configuration

- Training and evaluation scripts use Hydra configs; keep overrides explicit in CLI.
- Default config path in scripts: `config/` under the run module’s directory.
- Avoid hard-coding absolute paths; use env vars + config values.

### Torch / Lightning Conventions

- Keep model loading in `initialize()` (not `__init__`).
- `forward()` returns a dict of predictions; must include `"trajectory"`.
- Keep feature/target builders pure and deterministic for caching.

## Patterns in This Codebase

- Data flow:
  - `SceneLoader` -> `Dataset` -> feature/target builders -> model
- `AbstractAgent` defines the public agent API.
- Learning-based agents implement training hooks (`get_feature_builders`, `compute_loss`, etc.).
- Single-process scripts live in `navsim/planning/script/`.

## Cursor/Copilot Rules

- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` files found.

## Practical Tips for Agents

- Prefer editing within the existing file layout; add new agents under `navsim/agents/`.
- If adding training configs, keep them in the same pattern as `config/training`.
- Keep new scripts aligned with existing `run_*.py` style and logging usage.
- Avoid adding heavy dependencies unless required by a model or baseline.

## When Adding Tests (future)

- Use pytest naming: `test_*.py` and `*_test.py`.
- Keep tests deterministic; avoid requiring large datasets by default.
- Add lightweight unit tests for feature builders and utility helpers first.
