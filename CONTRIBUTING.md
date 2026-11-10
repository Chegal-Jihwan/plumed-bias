# Contributing

Development setup: follow the installation steps in the [README](README.md#installation), then run `pytest`.

For bugs, use the bug report template. Use a small synthetic structure; do not attach private structures or model checkpoints.

Changes to force coupling or restart behavior should include a reproducer and a regression check. Keep CV definitions for a particular system, model weights and machine-specific paths out of the core module.

The installed package is the single module `plumed_bias.py`; `examples/` and `benchmarks/` are not installed. CI builds the wheel on Linux CPU with the dependencies pinned in `ci/constraints-linux.txt`, installs it, and runs the tests, the minimal example and a benchmark smoke test outside the checkout. CUDA, OPES restart and other platforms are covered only by the recorded runs in [validation](docs/validation.md). Say in a pull request which of these you ran.
