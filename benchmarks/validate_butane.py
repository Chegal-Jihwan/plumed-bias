"""n-butane OPES restart check with SevenNet on CUDA. Each MD branch runs in a new process.

SevenNet provides the model and PLUMED provides OPES; this script drives their interfaces and
compares checkpoints and trajectories. Settings and limits: butane_validation.json.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch_sim as ts
from ase.build import molecule
from ase.io import write
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits as U

from plumed_bias import EV_KJMOL, PlumedBias, load_checkpoint, resume_bias, save_checkpoint

ROOT = Path(__file__).resolve().parent


def require(ok, detail):
    """Acceptance check that `python -O` cannot strip, unlike assert."""
    if not ok:
        raise RuntimeError(f"validation failed: {detail}")


def atoms_list(count):
    at = molecule("trans-butane")
    require(at.get_chemical_formula() == "C4H10" and len(at) == 14, "structure is not n-butane")
    require(at.get_chemical_symbols()[:4] == ["C"] * 4, "first four atoms are not the carbons")
    at.center(vacuum=10.0)
    return [at.copy() for _ in range(count)]


def text(cfg, out, b, restart=None):
    rs = "" if restart is None else f" RESTART=YES STATE_RFILE={restart}"
    return ("UNITS LENGTH=A ENERGY=eV TIME=ps\nphi: TORSION ATOMS=1,2,3,4 NOPBC\n"
            f"opes: OPES_METAD_EXPLORE ARG=phi TEMP={cfg['temperature_K']} PACE={cfg['pace']} "
            f"BARRIER={cfg['barrier_kJ_mol'] / EV_KJMOL:.17g} SIGMA_MIN={cfg['sigma_min_rad']} "
            f"FILE={out}/KERNELS.{b} STATE_WFILE={out}/STATE.{b} STATE_WSTRIDE={cfg['pace']}{rs}\n"
            f"PRINT ARG=phi,opes.bias STRIDE={cfg['pace']} FMT=%.17g FILE={out}/COLVAR.{b}\n")


def dump(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def data_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def phase(cfg, name, out):
    from sevenn.torchsim import SevenNetModel
    out.mkdir(parents=True, exist_ok=False)
    count = 2 if name == "batch" else 1
    atoms = atoms_list(count)
    write(out / "initial.extxyz", atoms)
    torch.manual_seed(cfg["seed"])
    base = SevenNetModel(cfg["model"], modal=cfg["modal"], device="cuda", dtype=torch.float32, enable_cueq=True)
    source = out.parent / "straight"
    restarting = name in ("resume", "no_rng")
    inputs = [text(cfg, out, b, source / f"STATE.{b}.checkpoint" if restarting else None) for b in range(count)]
    pb = PlumedBias(base, atoms, inputs, cfg["dt_fs"], cfg["temperature_K"] * U.temperature,
                    out, flush_stride=cfg["pace"])
    record = {"phase": name, "systems": count, "device": torch.cuda.get_device_name(),
              "versions": {p: importlib.metadata.version(p)
                           for p in ("torch", "torch-sim-atomistic", "sevenn", "ase")}}
    snapshots = {}
    ck, end = cfg["checkpoint_step"], cfg["end_step"]
    start = ck if restarting else 0
    if name == "smoke":
        end = cfg["smoke_steps"]
    if name == "batch":
        end = cfg["batch_steps"]
    if name == "no_rng":
        end = ck + cfg["negative_control_steps"]
    marks = {ck + 1, ck + 10, ck + 100, ck + 1000, ck + cfg["negative_control_steps"], cfg["end_step"]}
    try:
        if restarting:
            st, info = load_checkpoint(source / "checkpoint.pt", atoms, device="cuda", dtype=torch.float32)
            saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
            record["checkpoint_tensor_match"] = all(torch.equal(getattr(st, k).cpu(), saved[k])
                                                    for k in ("positions", "momenta", "forces", "energy"))
            record["rng_match_before_control"] = torch.equal(st.rng.get_state(), saved["rng"])
            require(record["checkpoint_tensor_match"] and record["rng_match_before_control"],
                    "checkpoint tensors or RNG differ")
            resume_bias(pb, info)
            require(pb.step == ck + 1, "bias step after resume")
            if name == "no_rng":
                st.rng = cfg["seed"] + 999
        else:
            st = ts.io.atoms_to_state(atoms, device="cuda", dtype=torch.float32)
            st.rng = cfg["seed"]
            st = nvt_langevin_init(st, pb, kT=cfg["temperature_K"] * U.temperature)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        measured_from = start
        for step in range(start + 1, end + 1):
            st = nvt_langevin_step(st, pb, dt=cfg["dt_fs"] * 1e-3 * U.time,
                                   kT=cfg["temperature_K"] * U.temperature,
                                   gamma=1 / (cfg["friction_time_ps"] * U.time))
            if name == "smoke" and step == 100:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                measured_from = step
            if step % cfg["pace"] == 0 or step in marks:
                for k in ("positions", "momenta", "forces", "energy"):
                    if not torch.isfinite(getattr(st, k)).all():
                        raise RuntimeError(f"non-finite {k} at step {step}")
                if not np.isfinite(pb.last_bias).all():
                    raise RuntimeError("non-finite bias")
            if step in marks:
                for key, value in (("x", st.positions), ("f", st.forces), ("e", st.energy)):
                    snapshots[f"{step}_{key}"] = value.detach().cpu().numpy().copy()
                snapshots[f"{step}_bias"] = pb.last_bias.copy()
            if name == "straight" and step == ck:
                save_checkpoint(out / "checkpoint.pt", st, pb, extra={"config": cfg})
                state = out / "STATE.0"
                if not state.is_file() or not data_rows(state):
                    raise RuntimeError("OPES checkpoint has no kernel data")
                shutil.copy2(state, out / "STATE.0.checkpoint")
                record["state_checkpoint_sha256"] = hashlib.sha256(state.read_bytes()).hexdigest()
            if step % 1000 == 0:
                dump(out / "progress.json",
                     {"phase": name, "step": step, "end": end, "next_bias_step": pb.step})
        torch.cuda.synchronize()
        record.update(steps_completed=end - start, last_step=end, next_bias_step=pb.step,
                      wall_measured_s=time.perf_counter() - tick)
        record["ms_per_step"] = 1e3 * record["wall_measured_s"] / (end - measured_from)
        require(pb.step == end + 1, "bias step at end")
        np.savez(out / "snapshots.npz", **snapshots)
    finally:
        pb.finalize()
    if name != "smoke":
        record["kernels_in_state"] = [len(data_rows(out / f"STATE.{b}")) for b in range(count)]
        require(min(record["kernels_in_state"]) > 0, 'min(record["kernels_in_state"]) > 0')
    dump(out / "result.json", record)
    print(json.dumps(record), flush=True)


def batch_isolation(cfg, root):
    atoms = atoms_list(2)
    x = np.concatenate([a.positions for a in atoms])
    y = x.copy()
    y[14 + 3, 2] += 0.3
    values = []
    for label, positions in (("unchanged", x), ("other_system_changed", y)):
        out = root / ("probe_" + label)
        out.mkdir()
        pb = PlumedBias(None, atoms, [text(cfg, out, b, root / "batch" / f"STATE.{b}") for b in range(2)],
                        cfg["dt_fs"], cfg["temperature_K"] * U.temperature, out)
        try:
            pb.step = cfg["batch_steps"] + 1
            values.append(pb.bias(positions))
        finally:
            pb.finalize()
    first_gap = max(float(abs(values[0][1][0] - values[1][1][0])),
                    float(np.abs(values[0][0][:14] - values[1][0][:14]).max()))
    require(first_gap < 1e-12, first_gap)
    return {"first_system_bias_force_gap_when_other_changes": first_gap}


def analyze(cfg, out):
    ref, rep, res, neg = [np.load(out / name / "snapshots.npz") for name in ("straight", "repeat", "resume", "no_rng")]
    first = cfg["checkpoint_step"] + 1
    gap = lambda a, b, step, key: float(np.abs(a[f"{step}_{key}"] - b[f"{step}_{key}"]).max())
    first_gaps = {key: gap(res, ref, first, key) for key in ("x", "f", "bias")}
    limits = cfg["first_step_limits"]
    require(first_gaps["x"] <= limits["positions_A"], first_gaps)
    require(first_gaps["f"] <= limits["forces_eV_A"], first_gaps)
    require(first_gaps["bias"] <= limits["bias_eV"], first_gaps)
    step = cfg["checkpoint_step"] + cfg["negative_control_steps"]
    negative, resumed = gap(neg, ref, step, "x"), gap(res, ref, step, "x")
    require(negative > max(1e-3, 10 * resumed), {"negative": negative, "resumed": resumed})
    time_ck = cfg["checkpoint_step"] * cfg["dt_fs"] * 1e-3
    times = lambda name: [float(row[0]) for row in data_rows(out / name / "KERNELS.0")
                          if float(row[0]) > time_ck + 1e-9]
    require(times("straight") == times("resume"), "OPES update schedule differs after restart")
    result = {"status": "passed", "first_resumed_step_gaps": first_gaps,
              "positions_gap_at_1ps_after_restart_A": {"resumed": resumed, "without_rng": negative},
              "final_positions_gap_A": {"resumed_vs_straight": gap(res, ref, cfg["end_step"], "x"),
                                        "repeat_vs_straight": gap(rep, ref, cfg["end_step"], "x")},
              "kernel_updates_after_checkpoint": len(times("resume")),
              "batch_isolation": batch_isolation(cfg, out),
              "interpretation": "Endpoint drift is reported, not a bitwise GPU trajectory gate; "
                                "no FES convergence claim."}
    dump(out / "summary.json", result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    phases = ("smoke", "straight", "repeat", "resume", "no_rng", "batch")
    p.add_argument("--phase", default="all", choices=("all",) + phases)
    p.add_argument("--out", type=Path, default=ROOT / "gpu_results")
    args = p.parse_args()
    cfg = json.loads((ROOT / "butane_validation.json").read_text())
    if args.phase != "all":
        phase(cfg, args.phase, args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        for name in phases:
            remaining = cfg["gpu_wall_limit_s"] - (time.monotonic() - started) - 90
            if remaining <= 0:
                raise RuntimeError("gpu_wall_limit_s exhausted")
            with (args.out / f"{name}.log").open("w") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", name,
                                         "--out", str(args.out / name)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect its log (rc={result.returncode})")
            if name == "smoke":
                ms = json.loads((args.out / name / "result.json").read_text())["ms_per_step"]
                steps = (2 * cfg["end_step"] + cfg["end_step"] - cfg["checkpoint_step"]
                         + cfg["negative_control_steps"] + cfg["batch_steps"])
                estimate = steps * ms / 1e3 * 1.2 + 600
                dump(args.out / "time_estimate.json",
                     {"smoke_ms_per_step": ms, "remaining_estimate_s": estimate, "budget_remaining_s": remaining})
                if estimate > remaining:
                    raise RuntimeError("smoke timing projects beyond gpu_wall_limit_s; "
                                       "stopped before the long branches")
        analyze(cfg, args.out)
    except Exception as exc:
        dump(args.out / "status.json", {"status": "failed_or_incomplete", "reason": str(exc)})
        raise


if __name__ == "__main__":
    main()
