"""plumed_bias: a PLUMED bias for torch-sim models, plus MD checkpoint helpers.

PlumedBias wraps a torch-sim model (anything that maps a state to {"energy": [n_systems], "forces": [n_atoms, 3]}).
On each call it runs the base model, copies the positions to the CPU in float64, passes them to one PLUMED instance
per system through PLUMED's python `cmd` interface (the setStep / setPositions / setMasses / setForces / setVirial /
getBias commands that ase.calculators.plumed also uses), and adds the bias force and bias energy to the model output.
The integrator is unchanged; the bias is part of the model.

Units. The MD side is set to Angstrom, eV and fs (setMDLengthUnits, setMDEnergyUnits, setMDTimeUnits). The PLUMED input
has its own units: start each input with `UNITS LENGTH=A ENERGY=eV` and give energies in eV (RESTRAINT KAPPA in eV/A^2,
METAD HEIGHT in eV). Input times stay in ps unless the UNITS line also sets TIME.

Restart. PLUMED keeps the bias history in files and torch-sim keeps the MD state in the state object, so a restart needs
both. `save_checkpoint` stores positions, momenta, forces, energy, state extras, the random generator and the bias step
counter; `load_checkpoint` rebuilds the state from the same atoms. PLUMED is restarted from its own input (METAD with
`RESTART=YES` and `FILE=<copy of the HILLS file>`, OPES with `RESTART=YES` and `STATE_RFILE=<copy of the STATE file>`).
Copy the HILLS / STATE file at a step where PLUMED has written it to disk (see `flush_stride`). Not stored: the PLUMED
print files and any internal state of the base model, such as a neighbour-list cache, which is rebuilt on the first
call.

The checkpoint helpers need torch-sim >= 0.6 (state.rng); state extras need torch-sim >= 0.6.2. System membership,
species and masses must stay fixed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

# PLUMED's internal value of 1 eV in kJ/mol, the one UNITS ENERGY=eV uses. The CODATA 2018 value 96.48533212 would put a
# 2.6e-7 relative offset on every bias energy.
EV_KJMOL = 96.48530749925792
CHECKPOINT_FORMAT = 2
MASS_RTOL = 1e-7    # float32 round-trip of atomic masses


class PlumedBias(torch.nn.Module):
    """Base model plus PLUMED bias, one PLUMED instance per system.

    base         torch-sim model, or None to use `bias()` on its own
    atoms_list   one non-periodic ase.Atoms per system (atom count, species and masses, checked against the state)
    inputs       PLUMED input text per system (see the module docstring for the unit lines)
    dt_fs        MD time step in fs (PLUMED needs it for its time axis)
    kT_eV        thermal energy in eV (PLUMED's setKbT)
    logdir       existing directory for the written copy of each input (`plumed_in.<i>.dat`) and the PLUMED logs
                 (`plumed.<i>.log`)
    sync_timers  call torch.cuda.synchronize() at each timing boundary (when CUDA is available) so that `self.timings`
                 (seconds spent in model, d2h, plumed, h2d; always accumulated) is accurate
    pre_init     (command, value) pairs sent to every instance before `init` (for example MPI communicators)
    flush_stride if > 0 and the input has no FLUSH line, `FLUSH STRIDE=<n>` is added (before ENDPLUMED, if
                 present): PLUMED flushes its output files every n steps

    If a PLUMED call raises, the instances of earlier systems have already processed that step; create a new PlumedBias
    from the last checkpoint instead of calling this one again.
    """

    def __init__(self, base, atoms_list, inputs, dt_fs, kT_eV, logdir, sync_timers=False, pre_init=(), flush_stride=0):
        super().__init__()
        atoms_list, inputs = list(atoms_list), list(inputs)
        if not atoms_list or len(atoms_list) != len(inputs):
            raise ValueError("provide one PLUMED input per system, with at least one system")
        if any(len(a) == 0 for a in atoms_list):
            raise ValueError("empty systems are not supported")
        if any(a.get_pbc().any() for a in atoms_list):
            raise ValueError("PlumedBias supports non-periodic systems only; use NOPBC in the input")
        if not np.isfinite(dt_fs) or dt_fs <= 0 or not np.isfinite(kT_eV) or kT_eV <= 0:
            raise ValueError("dt_fs and kT_eV must be finite and positive")
        if isinstance(flush_stride, bool) or not isinstance(flush_stride, (int, np.integer)) or flush_stride < 0:
            raise ValueError("flush_stride must be a non-negative integer")
        if not Path(logdir).is_dir():
            raise ValueError("logdir must be an existing directory")
        pre_init = tuple(pre_init)                      # sent to every instance; a one-shot iterator would not be
        import plumed
        self.base = base
        self._offsets = np.cumsum([0] + [len(a) for a in atoms_list])
        self._masses = [np.ascontiguousarray(a.get_masses(), dtype=np.float64) for a in atoms_list]
        self._expected = {"system_idx": np.repeat(np.arange(len(atoms_list)), np.diff(self._offsets)),
                          "atomic_numbers": np.concatenate([a.numbers for a in atoms_list]),
                          "masses": np.concatenate(self._masses)}
        self._state_checks = {}
        self._plumed = []
        self._called = False                            # set by the first PLUMED calc; resume_bias needs False
        self.failed = False                             # set when a call raises mid-step; save_checkpoint refuses
        for b, (a, txt) in enumerate(zip(atoms_list, inputs)):
            lines = txt.splitlines()
            words = [line.split("#", 1)[0].split() for line in lines]
            keys = [w[1:2] if w and w[0].endswith(":") else w[:1] for w in words]    # action name after any label
            end = keys.index(["ENDPLUMED"]) if ["ENDPLUMED"] in keys else len(lines)    # PLUMED ignores the rest
            if flush_stride and ["FLUSH"] not in keys[:end]:
                lines.insert(end, f"FLUSH STRIDE={int(flush_stride)}")
                txt = "\n".join(lines) + "\n"
            f = Path(logdir) / f"plumed_in.{b}.dat"
            f.write_text(txt)
            p = plumed.Plumed()
            for k, v in pre_init:
                if v is None:
                    p.cmd(k)
                else:
                    p.cmd(k, v)
            p.cmd("setMDEngine", "python")
            p.cmd("setNatoms", len(a))
            p.cmd("setMDLengthUnits", 0.1)              # 1 A = 0.1 nm
            p.cmd("setMDEnergyUnits", EV_KJMOL)         # 1 eV in kJ/mol
            p.cmd("setMDTimeUnits", 0.001)              # 1 fs in ps
            p.cmd("setTimestep", float(dt_fs))
            p.cmd("setKbT", float(kT_eV))
            p.cmd("setPlumedDat", str(f))
            p.cmd("setLogFile", str(Path(logdir) / f"plumed.{b}.log"))
            p.cmd("init")
            self._plumed.append(p)
        self.step = 0                                   # PLUMED step of the next call; integrator init is step 0
        self._sync_timers = sync_timers
        self.timings = {"model": 0.0, "d2h": 0.0, "plumed": 0.0, "h2d": 0.0}
        self.last_bias = None                           # bias energies [n_systems] (eV) of the last forward call
        for k in ("device", "dtype", "compute_forces", "memory_scales_with"):  # read by torch-sim
            if hasattr(base, k):
                setattr(self, k, getattr(base, k))
        self.compute_stress = False  # no bias virial: a stress returned by the base model excludes the bias

    def _validate_state(self, state):
        if tuple(state.positions.shape) != (self._offsets[-1], 3):
            raise ValueError("state.positions must have shape (total atoms from atoms_list, 3)")
        # These fields do not change between MD steps. Skip the CPU copy when the tensor object and its
        # PyTorch version counter are the same as at the last check.
        for name in ("system_idx", "atomic_numbers", "masses", "pbc"):
            value = getattr(state, name)
            try:
                version = value._version
            except RuntimeError:  # inference tensors have no mutation counter: always check
                version = None
            old = self._state_checks.get(name)
            if version is not None and old is not None and old[0] is value and old[1] == version:
                continue
            actual = value.detach().cpu().numpy()
            if name == "pbc":
                valid = not actual.any()
            else:
                expected = self._expected[name]
                if name == "masses":
                    same = np.allclose(actual, expected, rtol=MASS_RTOL, atol=0)
                else:
                    same = np.array_equal(actual, expected)
                valid = actual.shape == expected.shape and same
            if not valid:
                raise ValueError(f"state.{name} does not match the non-periodic atoms_list layout")
            self._state_checks[name] = (value, version)

    def _tick(self):
        if self._sync_timers and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def bias(self, x):
        """Bias at the current `self.step`, which is not advanced.

        x: [N, 3] positions in A, all systems stacked.
        Returns (bias forces [N, 3] in eV/A, bias energies [n_systems] in eV). A non-finite result (for example a
        DISTANCE between overlapping atoms) finalizes the wrapper and raises FloatingPointError. Any error raised
        by PLUMED sets `self.failed`.
        """
        if not self._plumed:
            raise RuntimeError("PlumedBias was finalized")
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self._offsets[-1], 3) or not np.isfinite(x).all():
            raise ValueError("positions must be finite with shape (total atoms from atoms_list, 3)")
        fb, vb = np.zeros_like(x), np.zeros(len(self._plumed))
        self._called = True
        try:
            for b, p in enumerate(self._plumed):
                pos = np.ascontiguousarray(x[self._offsets[b]:self._offsets[b + 1]])
                f = np.zeros_like(pos)
                vir = np.zeros((3, 3))
                e = np.zeros(1)
                p.cmd("setStep", self.step)
                p.cmd("setMasses", self._masses[b])
                p.cmd("setPositions", pos)
                p.cmd("setForces", f)
                p.cmd("setVirial", vir)
                p.cmd("calc")
                p.cmd("getBias", e)
                fb[self._offsets[b]:self._offsets[b + 1]] = f
                vb[b] = e[0]
        except BaseException:
            self.failed = True    # earlier systems have already taken this step
            raise
        if not (np.isfinite(fb).all() and np.isfinite(vb).all()):
            self.failed = True
            self.finalize()    # the instances have already taken this step: the wrapper cannot continue
            raise FloatingPointError(f"PLUMED returned a non-finite bias at step {self.step}; wrapper finalized")
        return fb, vb

    def forward(self, state, **kw):
        """Base model output plus the PLUMED bias, without `stress`; advances the PLUMED step by one.

        If this raises, the integrator has usually updated the state in place already: the state is mid-step and
        `save_checkpoint` refuses it. Restart from the last checkpoint.
        """
        try:
            return self._forward(state, **kw)
        except BaseException:
            self.failed = True
            raise

    def _forward(self, state, **kw):
        if not self._plumed:
            raise RuntimeError("PlumedBias was finalized")
        self._validate_state(state)
        t0 = self._tick()
        res = self.base(state, **kw)
        t1 = self._tick()
        n_sys = len(self._plumed)
        if tuple(res["energy"].shape) != (n_sys,) or tuple(res["forces"].shape) != (self._offsets[-1], 3):
            raise ValueError(f"base model must return energy [{n_sys}] and forces [{self._offsets[-1]}, 3]")
        if not (torch.isfinite(res["energy"]).all() and torch.isfinite(res["forces"]).all()):
            raise FloatingPointError("base model returned non-finite energy or forces; the bias was not advanced")
        x = state.positions.detach().cpu().to(torch.float64).numpy()    # CPU first: MPS has no float64
        t2 = self._tick()
        fb, vb = self.bias(x)
        t3 = self._tick()
        F = torch.from_numpy(fb).to(res["forces"].device, res["forces"].dtype)
        V = torch.from_numpy(vb).to(res["energy"].device, res["energy"].dtype)
        out = dict(res)
        out.pop("stress", None)    # no bias virial, so a base-model stress would not be the biased stress
        out["forces"] = res["forces"] + F
        out["energy"] = res["energy"] + V
        t4 = self._tick()
        for k, dt in zip(("model", "d2h", "plumed", "h2d"), (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            self.timings[k] += dt
        self.last_bias = vb
        self.step += 1
        return out

    def finalize(self):
        """Close all PLUMED instances (writes the final files). Later calls raise; a second finalize does nothing."""
        for p in self._plumed:
            p.finalize()
        self._plumed = []


def _check_tensors(ck, n, n_sys):
    for k, shape in (("positions", (n, 3)), ("momenta", (n, 3)), ("forces", (n, 3)), ("energy", (n_sys,))):
        if tuple(ck[k].shape) != shape or not torch.isfinite(ck[k]).all():
            raise ValueError(f"checkpoint {k} must be finite with shape {shape}")
    for kind, lead in (("system_extras", n_sys), ("atom_extras", n)):
        for k, v in ck[kind].items():
            if not isinstance(v, torch.Tensor) or v.shape[:1] != (lead,):
                raise ValueError(f"checkpoint {kind} {k!r} must be a tensor with leading dimension {lead}")
            if v.is_floating_point() and not torch.isfinite(v).all():
                raise ValueError(f"checkpoint {kind} {k!r} must be finite")


def save_checkpoint(path, state, bias=None, extra=None):
    """Write the dynamic part of an MD state: positions, momenta, forces, energy, system/atom extras, the random
    generator and the bias step.

    With `bias` (the PlumedBias driving the run), its step counter is stored and the state is checked against its
    layout. A state whose last wrapper call raised is refused, since it may be mid-step; a cleanly finalized `bias` is
    fine. Wrong shapes and non-finite values are refused too, and the previous file is kept in all these cases.

    The file is written to `<path>.tmp`, read back and renamed, so an interrupted or failed write leaves the previous
    checkpoint in place. `extra` is a dict of tensors, numbers or strings, returned by `load_checkpoint`.
    """
    from torch_sim.integrators.md import MDState
    if type(state) is not MDState or state.constraints:
        raise ValueError("checkpoint helpers support unconstrained MDState only (NVE or Langevin NVT)")
    if bias is not None:
        if bias.failed:
            raise ValueError("a call of this bias raised, so the state may be mid-step; use the last checkpoint")
        bias._validate_state(state)
    ck = {"format": CHECKPOINT_FORMAT,
          "n_atoms": int(state.positions.shape[0]),
          "positions": state.positions.detach().cpu(), "momenta": state.momenta.detach().cpu(),
          "forces": state.forces.detach().cpu(), "energy": state.energy.detach().cpu(),
          "rng": state.rng.get_state().clone(),
          "rng_device": state.rng.device.type,
          "identity": {k: getattr(state, k).detach().cpu().clone()
                       for k in ("atomic_numbers", "masses", "system_idx", "pbc", "cell")},
          "bias_step": None if bias is None else int(bias.step),
          "extra": extra or {}}
    for kind in ("system_extras", "atom_extras"):          # torch-sim >= 0.6.2; may feed the model
        ck[kind] = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in getattr(state, kind, {}).items()}
    _check_tensors(ck, ck["n_atoms"], int(state.system_idx.max()) + 1)
    tmp = os.fspath(path) + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            torch.save(ck, fh)
            fh.flush()
            os.fsync(fh.fileno())             # on disk before the rename replaces the previous checkpoint
        torch.load(tmp, weights_only=True)    # fail here, not at load time, on an unreadable `extra`
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)


def load_checkpoint(path, atoms_list, device="cpu", dtype=None):
    """Rebuild the MD state saved by `save_checkpoint` from the same atoms (masses, species, cell and system layout).

    `dtype` defaults to the precision the checkpoint was saved in; a different one is refused.
    Returns (state, info) with info = {"bias_step", "extra", "positions"}. Call `resume_bias(bias, info)` before the
    next step.
    """
    import torch_sim as ts
    from torch_sim.integrators.md import MDState
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format {ck['format']}, this code reads {CHECKPOINT_FORMAT}")
    saved_dtype = ck["positions"].dtype
    if dtype is not None and dtype != saved_dtype:
        raise ValueError(f"checkpoint was saved in {saved_dtype}; omit dtype or pass dtype={saved_dtype}")
    st = ts.io.atoms_to_state(atoms_list, device=device, dtype=saved_dtype)
    if st.positions.shape[0] != ck["n_atoms"]:
        raise ValueError(f"checkpoint holds {ck['n_atoms']} atoms, atoms_list has {st.positions.shape[0]}")
    for kind in ("system_extras", "atom_extras"):
        ck.setdefault(kind, {})
        if ck[kind] and not hasattr(st, kind):
            raise ValueError(f"checkpoint holds {kind}, which this torch-sim version does not support")
    _check_tensors(ck, st.positions.shape[0], len(atoms_list))
    for k in ("atomic_numbers", "masses", "system_idx", "pbc", "cell"):
        saved = ck["identity"][k]
        actual = getattr(st, k).detach().cpu()
        expected = saved.to(actual)
        if actual.is_floating_point():
            same = torch.allclose(actual, expected, rtol=MASS_RTOL, atol=0)
        else:
            same = torch.equal(actual, expected)
        if actual.shape != expected.shape or not same:
            raise ValueError(f"checkpoint {k} does not match atoms_list (species, masses and layout must be unchanged)")
    if ck["rng_device"] != st.positions.device.type:
        raise ValueError("restore on the same device type as the checkpoint RNG (CPU and CUDA generators differ)")
    st.positions = ck["positions"].to(st.positions)     # saved positions replace the ones from atoms_list
    like = st.positions
    for kind in ("system_extras", "atom_extras"):
        for k, v in ck[kind].items():
            getattr(st, kind)[k] = v.to(like.device)
    state = MDState.from_state(st, momenta=ck["momenta"].to(like), energy=ck["energy"].to(like),
                               forces=ck["forces"].to(like))
    g = torch.Generator(device=state.positions.device)
    g.set_state(ck["rng"])
    state.rng = g
    return state, {"bias_step": ck["bias_step"], "extra": ck["extra"], "positions": ck["positions"]}


def resume_bias(bias, info):
    """Put the PLUMED step counter back to where the checkpoint was taken (the next call is the next step).

    `bias` must be a new PlumedBias that has not been called. A new METAD/OPES instance skips the history update of its
    first call, so this replays the last completed step from the checkpoint positions: that update is already in the
    HILLS/STATE snapshot, and the next MD step then updates as in an uninterrupted run. If that step is a PRINT step,
    the replay writes its line again to the new output files. If the replay fails, `bias` is finalized.
    Does nothing if the checkpoint was saved without a bias.
    """
    step = info["bias_step"]
    if step is None:
        return
    if bias._called:
        raise ValueError("resume_bias needs a new PlumedBias that has not been called yet")
    x = info["positions"].numpy()
    if x.shape != (bias._offsets[-1], 3):
        raise ValueError("checkpoint positions do not match the atoms of this PlumedBias")
    if step > 0:
        bias.step = step - 1
        try:
            bias.bias(x)
        except BaseException:
            bias.failed = True
            bias.finalize()    # some instances may have replayed: this wrapper cannot be resumed again
            raise
    bias.step = step
