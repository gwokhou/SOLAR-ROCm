# Repository Guidelines

## Layout

- Python package: `solar/`; shared code: `solar/common/`.
- Pipeline layers: graph extraction in `solar/graph/`, einsum conversion/handlers in `solar/einsum/`, analysis in `solar/analysis/`, performance models in `solar/perf/`, and CLI entry points in `solar/cli/`.
- Architecture configs: `configs/arch/`; examples: `examples/`; docs: `docs/`; tests: `tests/`.

## Environment and Commands

- Use Python 3.10+; install with `pip install -r requirements.txt` and `pip install -e .` (the `dev` extra adds tooling).
- Run `bash scripts/run_tests.sh all` for the full suite or `bash scripts/run_tests.sh quick` for smoke tests.
- Run a focused test with `python3 -m pytest tests/test_einsum_analyzer.py -v`; add `--cov=solar --cov-report=html` for coverage.
- The host GPU and default UV cache are available. If the sandbox blocks either, request elevation and retry before changing cache paths or diagnosing the host.

## Code and Tests

- Follow the Google Python Style Guide and local conventions: four spaces, `snake_case` functions/modules, and `PascalCase` classes.
- Preserve SPDX headers; format and lint changed Python with Black, pylint, and mypy.
- Add deterministic regression tests for every change, using `tests/conftest.py` fixtures where applicable. Run focused tests before broader suites and resolve all warnings/failures.

## Commits and PRs

- Start changes from an approved SOLAR issue and keep each PR focused.
- Use signed commits titled `#<issue> - <imperative summary>` (`git commit -s`).
- Document user-visible changes and testing. Prefix incomplete PRs with `[WIP]`; attach screenshots/output when useful.
