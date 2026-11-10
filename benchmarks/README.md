# Bridge overhead benchmark

This benchmark repeatedly evaluates fixed positions with a small harmonic torch model, then with the same model wrapped by PLUMED. The PLUMED arm adds a distance restraint. It times the bridge itself (positions out, bias forces back), not a real model or the integrator.

```bash
python benchmarks/benchmark_bridge.py \
  --device cpu --walkers 1 4 16 --calls 200 --warmup 20 --repeats 3 \
  --out bridge_benchmark.json

python -m pip install -e ".[benchmark]"
python benchmarks/plot_bridge.py bridge_benchmark.json --out bridge_benchmark.svg
```

Use `--device cuda` to include CPU/GPU transfers on a CUDA machine. The script synchronizes CUDA at timing boundaries, alternates arm order across repeats, and excludes PLUMED initialization and warm-up from the measured interval. Each system has its own PLUMED instance; there is no bias sharing.

## Output

The JSON stores the command, environment, settings and each repeat's raw milliseconds per batch call. It also reports mean and standard deviation, milliseconds per system call (batch time divided by system count), and system calls per second (1000 * system count / milliseconds per batch call). The plot shows mean +/- one standard deviation across repeats, divided by system count (3 repeats by default).

The output path must be new. Keep the JSON with any figure so the plotted values can be regenerated. A macOS CPU run is included: [JSON](results/macos_cpu.json), [figure](results/macos_cpu.svg).

## CUDA OPES integration case

`validate_butane.py` and `butane_validation.json` are the CUDA OPES restart harness; the protocol and the recorded result are in [validation](../docs/validation.md#n-butane-cuda--opes-restart-2026-11-07).
