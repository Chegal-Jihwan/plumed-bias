"""Run independent non-periodic replicas with SevenNet and a PLUMED input file.

The built-in water structure is an integration demo. Choose and validate a model,
structure, time step and CVs for your research before using the example for sampling.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch_sim as ts
from ase.build import molecule
from ase.io import read, write
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits as U

from plumed_bias import PlumedBias, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path,
                        help="one ASE-readable non-periodic structure; default: built-in water")
    parser.add_argument("--model", default="7net-omni", help="SevenNet model name or checkpoint path")
    parser.add_argument("--modal", default="omol25_low",
                        help="model fidelity; use an empty string for a single-modal model")
    parser.add_argument("--plumed", type=Path,
                        default=Path(__file__).parent / "inputs/distance_metad.dat")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cueq", action="store_true", help="optional cuEquivariance backend, CUDA only")
    parser.add_argument("--walkers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--checkpoint-stride", type=int, default=20,
                        help="PLUMED flush interval in steps; --steps must be a multiple of it. "
                             "Match the METAD PACE or OPES STATE_WSTRIDE")
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("sevennet_run"),
                        help="new output directory, without whitespace")
    args = parser.parse_args()
    if args.walkers < 1 or args.steps < 1 or not np.isfinite(args.dt_fs) or args.dt_fs <= 0:
        parser.error("walkers, steps and dt-fs must be positive")
    if args.checkpoint_stride < 1 or args.steps % args.checkpoint_stride:
        parser.error("steps must be a multiple of the positive checkpoint-stride")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    if args.cueq and args.device != "cuda":
        parser.error("--cueq requires --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    out = args.out.resolve()
    if out.exists():
        parser.error("output directory already exists; choose a new --out path")
    if any(c.isspace() for c in str(out)):
        parser.error("choose an output path without whitespace for the PLUMED FILE templates")
    atoms = read(args.structure) if args.structure else molecule("H2O")
    if atoms.pbc.any():
        parser.error("this example supports non-periodic structures only")
    if len(atoms) < 2:
        parser.error("the distance input requires at least two atoms")
    if atoms.constraints:
        parser.error("the checkpoint example supports unconstrained structures only")
    # The base model may still compute a stress internally (the wrapper drops it); keep its cell volume finite.
    if abs(np.linalg.det(atoms.cell.array)) < 1e-12:
        atoms.center(vacuum=10.0)                       # A
    systems = [atoms.copy() for _ in range(args.walkers)]
    template = args.plumed.read_text()
    inputs = [template.replace("@OUT@", str(out)).replace("@WALKER@", str(b))
              .replace("@TEMP@", str(args.temperature)) for b in range(args.walkers)]
    try:
        from sevenn.torchsim import SevenNetModel
    except ImportError as exc:
        raise SystemExit('Install the optional model dependencies with: pip install ".[sevennet]"') from exc
    base = SevenNetModel(model=args.model, modal=args.modal or None, device=args.device,
                         dtype=torch.float32, enable_cueq=args.cueq)
    out.mkdir(parents=True, exist_ok=False)  # never append to an earlier run's histories
    write(out / "initial.extxyz", systems)
    kT = args.temperature * U.temperature
    dt = args.dt_fs * 1e-3 * U.time
    gamma = 1.0 / (0.1 * U.time)
    state = ts.io.atoms_to_state(systems, device=args.device, dtype=torch.float32)
    state.rng = args.seed
    pb = PlumedBias(base, systems, inputs, args.dt_fs, kT, out, flush_stride=args.checkpoint_stride)
    try:
        state = nvt_langevin_init(state, pb, kT=kT)  # PLUMED step 0
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            state = nvt_langevin_step(state, pb, dt=dt, kT=kT, gamma=gamma)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        pb.finalize()  # also closes output files if integration raises
    # The default METAD history has now been flushed and closed. Custom OPES inputs
    # also need an up-to-date STATE_WFILE; see docs/usage.md before restarting.
    save_checkpoint(out / "checkpoint.pt", state, pb)
    write(out / "final.extxyz", ts.io.state_to_atoms(state))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    model_path = Path(args.model)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None
    record = {"config": config, "steps": args.steps, "next_bias_step": pb.step,
              "wall_s": elapsed, "final_energy_with_bias_eV": state.energy.detach().cpu().tolist(),
              "plumed_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "model_checkpoint_sha256": model_sha256,
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "torch-sim-atomistic", "sevenn", "ase")},
              "command": sys.argv, "dtype": "float32", "thermostat_time_ps": 0.1,
              "note": "integration example; these timings include warm-up effects and are not a benchmark"}
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Completed {args.steps} steps for {args.walkers} independent systems.")
    print(f"Output: {out}")
    print("Files: initial.extxyz, final.extxyz, checkpoint.pt, run.json, HILLS.<i>, COLVAR.<i>, plumed.<i>.log")


if __name__ == "__main__":
    main()
