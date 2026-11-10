"""Minimal use of plumed_bias with torch-sim: a toy harmonic model, a distance restraint plus well-tempered
metadynamics, a checkpoint at step 200 and a restart from it.
Replace HarmonicToy by a real torch-sim model (for example SevenNetModel).
run from the package root: python examples/minimal_example.py
(needs torch-sim >= 0.6, ase, the PLUMED python module and kernel)"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase import Atoms
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits as U

from plumed_bias import PlumedBias, load_checkpoint, resume_bias, save_checkpoint

# two independent systems
atoms = [Atoms("C6", positions=np.random.default_rng(s).uniform(0, 3.0, (6, 3))) for s in (1, 2)]
# kT at 300 K in eV (torch-sim's U.temperature differs from this constant by 3e-7), 1 fs step,
# Langevin friction 1/(0.1 ps), METAD PACE in steps
kT, dt, gamma, PACE = 300 * 8.617333262e-5, 1e-3 * U.time, 1.0 / (0.1 * U.time), 20


class HarmonicToy(torch.nn.Module):
    """E = 0.5 k |x - x_ref|^2 per system (stand-in for a real model)."""

    def __init__(self, ref, k=2.0):
        super().__init__()
        self.register_buffer("ref", torch.as_tensor(ref, dtype=torch.float32))
        self.k, self.device, self.dtype = k, torch.device("cpu"), torch.float32
        self.compute_forces, self.compute_stress, self.memory_scales_with = True, False, "n_atoms"

    def forward(self, state, **kw):
        d = state.positions - self.ref
        e = torch.zeros(int(state.system_idx.max()) + 1)
        e.index_add_(0, state.system_idx, 0.5 * self.k * (d ** 2).sum(1))
        return {"energy": e, "forces": -self.k * d}


def plumed_input(out, b, restart_file=None):
    hills, flag = (f"{out}/HILLS.{b}", "") if restart_file is None else (str(restart_file), " RESTART=YES")
    return "\n".join(["UNITS LENGTH=A ENERGY=eV",                       # input units: Angstrom, eV (times stay in ps)
                      "d: DISTANCE ATOMS=1,2 NOPBC",
                      "rst: RESTRAINT ARG=d AT=1.5 KAPPA=3.0",
                      f"metad: METAD ARG=d SIGMA=0.1 HEIGHT=0.02 PACE={PACE} BIASFACTOR=8 TEMP=300 FILE={hills}{flag}",
                      f"PRINT ARG=d,metad.bias STRIDE=10 FILE={out}/COLVAR.{b}"]) + "\n"


def model(out, restart_files=None):
    ref = np.concatenate([a.get_positions() for a in atoms])
    texts = [plumed_input(out, b, None if restart_files is None else restart_files[b]) for b in range(2)]
    # flush_stride: HILLS on disk at checkpoint time
    return PlumedBias(HarmonicToy(ref), atoms, texts, dt_fs=1.0, kT_eV=kT, logdir=out, flush_stride=PACE)


def run(state, pb, a, b):
    for _ in range(a, b):
        state = nvt_langevin_step(state, pb, dt=dt, kT=kT, gamma=gamma)
    return state


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out1, out2 = Path(tmp, "run1"), Path(tmp, "run2")
        out1.mkdir(); out2.mkdir()
        pb = model(out1)
        try:
            state = ts.io.atoms_to_state(atoms, device="cpu", dtype=torch.float32)
            state.rng = 7  # seeds the thermostat noise and the initial momenta
            state = run(nvt_langevin_init(state, pb, kT=kT), pb, 0, 200)
            save_checkpoint(Path(tmp, "ckpt.pt"), state, pb)
            hills = [shutil.copy(out1 / f"HILLS.{b}", out2 / f"HILLS.{b}.restart") for b in range(2)]
            state = run(state, pb, 200, 400)
        finally:
            pb.finalize()
        print("uninterrupted, 400 steps: first position", state.positions[0].tolist())

        pb2 = model(out2, hills)
        try:
            state2, info = load_checkpoint(Path(tmp, "ckpt.pt"), atoms)
            resume_bias(pb2, info)
            state2 = run(state2, pb2, 200, 400)
        finally:
            pb2.finalize()
        print("restarted at 200, 400 steps: first position", state2.positions[0].tolist())
        gap = float((state.positions - state2.positions).abs().max())
        print("max |difference| (A):", gap)
        if not gap < 1e-5:                                # also fails on NaN
            raise RuntimeError("CPU restart differs from the uninterrupted run")


if __name__ == "__main__":
    main()
