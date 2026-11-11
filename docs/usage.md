# Using the wrapper

## Model and state contract

The base model accepts a torch-sim state and returns `energy` with shape `[B]` in eV and `forces` with shape `[N, 3]` in eV/Angstrom. It must expose the device/dtype properties required by the torch-sim integrator. `PlumedBias` runs the base model, copies positions to CPU float64, evaluates one PLUMED instance per system, and adds the returned bias energy and forces on the original output device.

Use a non-empty sequence of ASE Atoms, one PLUMED input per system and an existing output directory. Systems must be non-periodic. Atoms are stacked in the same order as `atoms_list`; membership, species and masses remain fixed. The wrapper checks the state layout whenever those tensors are replaced or edited in place through PyTorch, and raises `ValueError` if system membership, species, masses or periodicity changed. Writes through `.data` or a NumPy view are not detected.

The wrapper passes positions and masses to PLUMED, not the base model energy, charges, cell or virial. Actions that read them, such as ENERGY, get zeros, and the wrapper does not warn.

Use `UNITS LENGTH=A ENERGY=eV` at the start of each PLUMED input and `NOPBC` for geometric CVs. Times in the input (for example the HILLS time column or OPES/METAD time constants) stay in ps unless `TIME` is also set. `dt_fs` (fs) and `kT_eV` (eV) are passed to PLUMED as the time step and setKbT. With these units RESTRAINT KAPPA is in eV/Angstrom^2.

## Calls advance the bias

The supported Langevin example makes one model call during initialization (PLUMED step 0), then one call per MD step. Calling the wrapper for a diagnostic force evaluation also advances the bias and may deposit a hill. For logging, use the state returned by the step function and `pb.last_bias` instead of calling the wrapper again. Integrators that evaluate forces more than once per MD step (for example multiple-time-step or predictor-corrector schemes) advance the counter more than once per step and are not supported with history-dependent bias.

The lower-level `bias(x)` accepts stacked NumPy positions and uses the current `step`; it does not increment it.

## Closing output files

Always close the wrapper, including on integration errors:

```python
pb = PlumedBias(base_model, atoms_list, inputs, dt_fs, kT_eV, logdir)
try:
    # Initialize and integrate with pb here.
    ...
finally:
    pb.finalize()
```

Finalized objects cannot be reused. A finalized wrapper can still be passed to `save_checkpoint`.

If PLUMED raises during a call, including a direct `pb.bias(x)`, the instances of earlier systems have already taken that step: the wrapper sets `pb.failed` and needs to be replaced. If the base model returns NaN or inf, the wrapper raises `FloatingPointError` and the step is not advanced. If PLUMED returns a non-finite bias, for example for a DISTANCE between overlapping atoms, the wrapper raises and finalizes itself, and the HILLS or STATE file may contain that step.

The integrator updates the state in place, so after any exception from a wrapper call the state is usually half a step along. The wrapper sets `pb.failed`, and `save_checkpoint(path, state, pb)` refuses that state. Restart from the last checkpoint and its file copies.

The wrapper has no bias virial: it declares `compute_stress=False` and drops any `stress` the base model returns, since that value would exclude the bias.

## Restart: two parts

The MD checkpoint and PLUMED history must describe the same completed step.

1. Use `flush_stride` (usually METAD PACE) and save at a matching flush step. For OPES, configure `STATE_WFILE` and an appropriate `STATE_WSTRIDE`; flushing print buffers does not create a missing OPES state snapshot.
2. Call `save_checkpoint(path, state, pb)` and copy the HILLS or STATE file for that checkpoint. Keep this snapshot separate from a history that will continue growing.
3. Build a new wrapper with METAD `RESTART=YES FILE=<HILLS snapshot>` or OPES `RESTART=YES STATE_RFILE=<STATE snapshot>` and new output paths.
4. Call `load_checkpoint(path, atoms_list, device=...)`. The state comes back in the dtype it was saved in; passing a different `dtype=` raises.
5. Call `resume_bias(pb, info)` on the new wrapper before any other call, including `pb.bias`. If it raises, the wrapper is finalized and you need a new one.
6. Continue with the step function. Do not call the Langevin initialization again; the momenta, forces and RNG come from the checkpoint.

A new PLUMED METAD or OPES instance skips the history update on its first call. `resume_bias` spends that call on the last completed step, using the checkpoint positions, so the next step updates as in an uninterrupted run whether or not it is a deposition step. If the checkpoint step is a PRINT step, its line is written again to the new output files.

The HILLS or STATE file carries the deposited history, not every internal estimator. METAD with `ADAPTIVE=DIFF` keeps a running CV-fluctuation estimate outside the HILLS file, and the restarted run deposits hills with different widths. Exact restart is tested for fixed-width METAD; the adaptive-width difference is recorded in [validation](validation.md#plumed-details).

Exact restart is also tested for OPES_METAD and OPES_METAD_EXPLORE with fixed and adaptive SIGMA, on a kernel that has the opes module.

The CPU [complete example](../examples/minimal_example.py) implements this sequence and compares the continued trajectory with an uninterrupted one.

`save_checkpoint` refuses wrong shapes and non-finite positions, momenta, forces, energy or extras before the previous file is replaced. System and atom extras of the torch-sim state (torch-sim >= 0.6.2) are saved and restored.

The checkpoint verifies species, masses, system membership, PBC and cell. Use the same model, atom order, dtype, device type, temperature, friction and time step when reproducing a run. CPU and CUDA RNG states are not interchangeable. Model weights and PLUMED inputs are not checked; keep them with the run. Extra thermostat/barostat state, constraints, model caches and print files are not saved.

## Recorded OPES restart case

The [CUDA n-butane harness](../benchmarks/validate_butane.py) exercises OPES_METAD_EXPLORE with an adaptive torsion width, a copied STATE snapshot and a fresh process loading the MD checkpoint. Its [result record](../benchmarks/results/butane_opes_restart.json) includes the configured limits and actual differences. The harness uses CUDA and `enable_cueq=True`. It needs a CUDA 13 build of torch, SevenNet 0.13.0 with its `cueq13` extra, and a PLUMED kernel built with OPES:

```bash
python -m pip install "sevenn[cueq13]==0.13.0"
python benchmarks/validate_butane.py --out butane_validation
```

pytest does not run it. The harness stops after the smoke run if the projected time exceeds `gpu_wall_limit_s` (3 h) in `butane_validation.json`. The output directory must be new.

## SevenNet example

Install the optional adapter dependencies into a suitable torch environment:

```bash
python -m pip install -e ".[sevennet]"
python examples/sevennet_md.py --device cpu --steps 40 --out water_run
```

The default is two independent copies of ASE's built-in water molecule, a SevenNet model (`7net-omni`, `omol25_low`) and a distance restraint plus METAD. It shows how the pieces connect and what the output directory contains. `--cueq` enables SevenNet's optional cuEquivariance backend when installed and running on CUDA.

To use your own inputs:

```bash
python examples/sevennet_md.py \
  --structure molecule.xyz --model /path/to/checkpoint.pth --modal omol25_low \
  --plumed my_input.dat --device cuda --walkers 4 --steps 1000 --out my_run
```

Use `--modal ""` for a single-modal checkpoint. The example accepts one structure, copied into independent replicas. A periodic structure raises `ValueError`. In custom PLUMED inputs, `@OUT@`, `@WALKER@` and `@TEMP@` are replaced with the output directory, replica index and temperature; PLUMED's own braces are preserved. Choose output paths without whitespace.

The output directory must be new. It contains initial/final structures, model and integration settings in `run.json`, the MD checkpoint, and per-system PLUMED inputs/logs/histories.

- `--checkpoint-stride` (default 20, the METAD PACE of the supplied input) is the PLUMED flush interval. Set it to your METAD PACE or OPES `STATE_WSTRIDE`; `--steps` must be a multiple of it.
- The checkpoint is written once, after the last step, when the METAD files have been flushed and closed. A custom OPES input must also write its STATE file at that step.
- The elapsed time in `run.json` includes warm-up.

## Platform notes

Set `PLUMED_KERNEL` to the kernel library matching the intended PLUMED installation. The Python binding and kernel versions are separate; include both when reporting problems. An OPES input also requires a kernel built with that module.

On the tested macOS environment, torch and conda-forge PLUMED loaded different OpenMP runtimes and aborted with `OMP: Error #15`. The tests there ran with `KMP_DUPLICATE_LIB_OK=TRUE`, which Intel's runtime documents as unsafe; use it for tests only. Linux does not need it.

In the PLUMED 2.9.2 builds tested here, more than one METAD walker with file sharing (`WALKERS_N` > 1) in one Python process crashed at the first `calc`. The supported default is independent histories. Shared OPES requires MPI and is outside the batched wrapper's supported configuration.
