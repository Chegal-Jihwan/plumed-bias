"""Measure force-call overhead with a synthetic harmonic model and a distance restraint.

Positions are fixed (no MD), so this measures the cost of the call, not MLIP speed or
sampling. The numbers hold for the recorded environment only.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import tempfile
import time

import numpy as np
import torch
import torch_sim as ts
from ase import Atoms

from plumed_bias import PlumedBias


class Harmonic(torch.nn.Module):
    def __init__(self, positions):
        super().__init__()
        self.register_buffer("reference", positions.clone())
        self.device, self.dtype = positions.device, positions.dtype
        self.compute_forces, self.compute_stress, self.memory_scales_with = True, False, "n_atoms"

    def forward(self, state):
        delta = state.positions - self.reference
        n_systems = state.cell.shape[0]
        energy = torch.zeros(n_systems, device=self.device, dtype=self.dtype)
        energy.index_add_(0, state.system_idx, (delta * delta).sum(1))
        return {"energy": energy, "forces": -2 * delta}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--walkers", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--atoms", type=int, default=6)
    p.add_argument("--calls", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("bridge_benchmark.json"))
    args = p.parse_args()
    if args.atoms < 2 or min(args.walkers + [args.calls, args.repeats]) < 1 or args.warmup < 0:
        p.error("use at least two atoms, positive walkers/calls/repeats, and non-negative warmup")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is unavailable")
    if args.out.exists():
        p.error("output exists; choose a new --out path")
    torch.set_num_threads(1)                        # keep the toy model off a thread pool

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    results = []
    kernel_version = None
    for b in args.walkers:
        atoms = [Atoms(f"C{args.atoms}", positions=np.random.default_rng(i).uniform(0, 3, (args.atoms, 3)))
                 for i in range(b)]
        state = ts.io.atoms_to_state(atoms, device=args.device, dtype=torch.float32)
        base = Harmonic(state.positions)
        times = {"base": [], "plumed": []}
        for repeat in range(args.repeats):
            with tempfile.TemporaryDirectory() as tmp:
                inputs = ["UNITS LENGTH=A ENERGY=eV\nd: DISTANCE ATOMS=1,2 NOPBC\n"
                          "rst: RESTRAINT ARG=d AT=1 KAPPA=1\n" for _ in atoms]
                kT = 300 * 8.617333262e-5   # eV at 300 K (torch-sim's U.temperature differs by 3e-7)
                pb = PlumedBias(base, atoms, inputs, 1.0, kT, tmp)
                try:
                    # Alternate order to reduce systematic warm-up/thermal drift bias.
                    order = ("base", "plumed") if repeat % 2 == 0 else ("plumed", "base")
                    for name in order:
                        model = base if name == "base" else pb
                        for _ in range(args.warmup):
                            model(state)
                        sync()
                        start = time.perf_counter()
                        for _ in range(args.calls):
                            model(state)
                        sync()
                        times[name].append(1e3 * (time.perf_counter() - start) / args.calls)
                finally:
                    pb.finalize()
                match = re.search(r"Version:\s+(\S+)", (Path(tmp) / "plumed.0.log").read_text())
                kernel_version = match.group(1) if match else None
        for name, values in times.items():
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            results.append({"arm": name, "systems": b, "atoms_per_system": args.atoms,
                            "ms_per_batch_call_mean": mean, "ms_per_batch_call_stdev": stdev,
                            "ms_per_system_call_mean": mean / b, "system_calls_per_s": 1000 * b / mean,
                            "repeat_ms_per_batch_call": values})
    report = {"workload": "fixed-position synthetic harmonic model, optional PLUMED distance restraint",
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": {"python": platform.python_version(), "platform": platform.system(),
                              "machine": platform.machine(), "device": args.device,
                              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                              "torch_threads": torch.get_num_threads(),
                              "plumed_kernel_version": kernel_version,
                              "plumed_binding_version": importlib.metadata.version("plumed"),
                              "versions": {p: importlib.metadata.version(p)
                                           for p in ("torch", "torch-sim-atomistic", "ase", "numpy")}},
              "command": sys.argv, "results": results,
              "limits": "No MLIP, no integration, no METAD/OPES history, no sampling-efficiency claim. "
                        "Base and PLUMED arms return different forces."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.out}")
    for row in results:
        print(f"{row['arm']:6} B={row['systems']:3}: {row['ms_per_system_call_mean']:.4f} ms/system-call")


if __name__ == "__main__":
    main()
