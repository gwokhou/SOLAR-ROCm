# Solar scripts

The maintained repository script is `run_tests.sh`. Historical external
wrappers such as `run_kernelbench*.sh` and `collect_perf_results.py` are not
part of this ROCm repository; use the Python CLI entry points instead.

## Test runner

Run the pinned ROCm environment setup once from the repository root:

```bash
bash install_uv.sh
```

Then use:

```bash
bash scripts/run_tests.sh quick
bash scripts/run_tests.sh all
bash scripts/run_tests.sh examples
bash scripts/run_tests.sh graph
bash scripts/run_tests.sh einsum
bash scripts/run_tests.sh integration
```

The runner prefers `.venv/bin/python`, never mutates the environment, and
returns a nonzero status for failed tests or examples.

## Pipeline and GPU evaluation

Use the installed CLIs for the five-stage analysis pipeline; see
[`docs/USAGE.md`](../docs/USAGE.md). Use `solar-evaluate` for executable ROCm
benchmarks; see [`docs/ROCM_BENCHMARKING.md`](../docs/ROCM_BENCHMARKING.md).

The default profile is `RX_9060_XT`. Architecture loading and executable
evaluation are both ROCm/AMD-only.
