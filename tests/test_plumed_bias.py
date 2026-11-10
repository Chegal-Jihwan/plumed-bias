"""CPU tests for plumed_bias.py with a toy harmonic model (no MLIP, no GPU). docs/validation.md lists each test.

Needs the package dependencies plus the PLUMED python module and kernel. test_restart_without_rng_diverges is the
control for the restart tests: without the generator state the run must differ.
run: python tests/test_plumed_bias.py   (or pytest tests/)   env: PLUMED_KERNEL, PYTHONPATH with the plumed module
"""
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase import Atoms
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits as U

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plumed_bias import PlumedBias, load_checkpoint, resume_bias, save_checkpoint  # noqa: E402

KB = 8.617333262e-5                         # eV/K; torch-sim's U.temperature differs from this by 3e-7 (relative)
KT = 300.0 * KB
PACE, GAMMA_BF = 20, 8.0
# PLUMED truncates its Gaussian kernels where 0.5*((s-c)/sigma)^2 reaches 6.25 and rescales (docs/validation.md)
E0 = float(np.exp(-6.25))
T = unittest.TestCase()


@contextmanager
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def toy_atoms(seed, n=6):
    return Atoms(f"C{n}", positions=np.random.default_rng(seed).uniform(0.0, 3.0, (n, 3)))


ATOMS = [toy_atoms(1), toy_atoms(2)]


def plumed_text(d, b, restart=None, pace=PACE):
    hills, flag = (f"{d}/HILLS.{b}", "") if restart is None else (str(restart), " RESTART=YES")
    return "\n".join(["UNITS LENGTH=A ENERGY=eV",
                      "d: DISTANCE ATOMS=1,2 NOPBC",
                      "rst: RESTRAINT ARG=d AT=1.5 KAPPA=3.0",
                      f"metad: METAD ARG=d SIGMA=0.1 HEIGHT=0.02 PACE={pace} BIASFACTOR={GAMMA_BF:g} TEMP=300 "
                      f"FILE={hills}{flag}",
                      f"PRINT ARG=d,rst.bias,metad.bias STRIDE=10 FMT=%.17g FILE={d}/COLVAR.{b}"]) + "\n"


def dist(x):
    return float(np.linalg.norm(x[0] - x[1]))


def read_hills(path):
    rows = [line.split() for line in open(path) if line.strip() and not line.startswith("#")]
    return np.array(rows, float).reshape(-1, 5)             # time, d, sigma_d, height, biasf


def metad_ref(hills, s):
    dp2 = 0.5 * ((s - hills[:, 1]) / hills[:, 2]) ** 2
    ker = np.where(dp2 < 6.25, (np.exp(-dp2) - E0) / (1 - E0), 0.0)
    # the HILLS height column is height * gamma / (gamma - 1)
    return float((hills[:, 3] * (hills[:, 4] - 1) / hills[:, 4] * ker).sum())


def rst_ref(s):
    return 0.5 * 3.0 * (s - 1.5) ** 2


def probe_steps():
    return (t for t in range(10_001, 10**6) if t % PACE)    # never a deposition / FLUSH step


def deposit(pb, n_steps, seed=3):
    rng = np.random.default_rng(seed)
    base = ATOMS[0].get_positions()
    for k in range(1, n_steps + 1):
        pb.step = k
        pb.bias(base + rng.normal(0, 0.15, base.shape))     # one walker; hills at k = 20, 40, ...
    return base


def one_walker(d, restart=None, flush_stride=0):
    d.mkdir(exist_ok=True)
    return PlumedBias(None, [ATOMS[0]], [plumed_text(d, 0, restart)], 1.0, KT, d, flush_stride=flush_stride)


def test_units_restraint():
    with tmpdir() as d:
        pb = one_walker(d)
        x = ATOMS[0].get_positions()
        pb.step = 1
        f, v = pb.bias(x)
        s = dist(x)
        u = (x[0] - x[1]) / s
        assert abs(v[0] - rst_ref(s)) < 1e-9, (v[0], rst_ref(s))
        assert np.abs(f[0] + 3.0 * (s - 1.5) * u).max() < 1e-9 and np.abs(f[1] - 3.0 * (s - 1.5) * u).max() < 1e-9
        assert np.abs(f[2:]).max() == 0.0
        pb.finalize()


def test_force_matches_finite_difference():
    with tmpdir() as d:
        pb = one_walker(d)
        x = deposit(pb, 100) + np.random.default_rng(9).normal(0, 0.1, (6, 3))
        steps, rng, gap = probe_steps(), np.random.default_rng(5), []
        pb.step = next(steps)
        f, _ = pb.bias(x)
        for _ in range(6):
            u = rng.normal(size=x.shape)
            u /= np.linalg.norm(u)
            pb.step = next(steps)
            _, vp = pb.bias(x + 1e-4 * u)
            pb.step = next(steps)
            _, vm = pb.bias(x - 1e-4 * u)
            gap.append(abs((vp[0] - vm[0]) / 2e-4 + float((f * u).sum())))
        assert max(gap) < 1e-7, max(gap)
        pb.finalize()


def test_metad_matches_numpy_reference_and_reload():
    with tmpdir() as d:
        pb = one_walker(d)
        x = deposit(pb, 100) + np.random.default_rng(9).normal(0, 0.1, (6, 3))
        pb.step = next(probe_steps())
        _, v_run = pb.bias(x)
        pb.finalize()
        hills = read_hills(d / "HILLS.0")
        assert len(hills) == 5, len(hills)                   # deposits at steps 20 .. 100
        s = dist(x)
        expected = metad_ref(hills, s) + rst_ref(s)
        assert abs(v_run[0] - expected) < 1e-8, (v_run[0], expected)
        shutil.copy(d / "HILLS.0", d / "HILLS.0.reload")
        pb2 = one_walker(d / "r", restart=d / "HILLS.0.reload")
        pb2.step = next(probe_steps())
        _, v_rel = pb2.bias(x)
        assert abs(v_rel[0] - v_run[0]) < 1e-8, (v_rel[0], v_run[0])
        pb2.finalize()


def test_systems_are_independent():
    with tmpdir() as d:
        pb = PlumedBias(None, ATOMS, [plumed_text(d, 0), plumed_text(d, 1)], 1.0, KT, d)
        x = np.concatenate([a.get_positions() for a in ATOMS])
        pb.step = 1
        _, v1 = pb.bias(x)
        y = x.copy()
        y[6:] += np.random.default_rng(4).normal(0, 0.2, (6, 3))   # a rigid shift would not change a distance
        pb.step = 2
        _, v2 = pb.bias(y)
        assert v1[0] == v2[0] and v1[1] != v2[1]
        pb.finalize()


def test_hills_on_disk_without_flush():
    """records whether METAD itself flushes a deposited hill to disk (the restart copy needs that or flush_stride)."""
    with tmpdir() as d:
        pb = one_walker(d)
        base = ATOMS[0].get_positions()
        pb.step = 1
        pb.bias(base)  # METAD never deposits on the first call it sees
        pb.step = 20
        pb.bias(base + 0.1)
        n_plain = len(read_hills(d / "HILLS.0"))
        print(f"  hills visible on disk right after a deposition, no FLUSH: {n_plain}")
        pb.finalize()
        with tmpdir() as d2:
            pb = one_walker(d2, flush_stride=20)
            pb.step = 1
            pb.bias(base)
            pb.step = 20
            pb.bias(base + 0.1)
            n_flush = len(read_hills(d2 / "HILLS.0"))
            print(f"  with flush_stride=20: {n_flush}")
            assert n_flush == 1
            pb.finalize()


def test_invalid_constructor_inputs_are_rejected_before_creating_files():
    with tmpdir() as d:
        periodic = ATOMS[0].copy()
        periodic.pbc = True
        cases = [([], []), (ATOMS, ["one input"]), ([Atoms()], ["input"]), ([periodic], ["input"])]
        for atoms, texts in cases:
            with T.assertRaises(ValueError):
                PlumedBias(None, atoms, texts, 1.0, KT, d)
        for kw in ({"dt_fs": 0}, {"dt_fs": np.nan}, {"kT_eV": -1}, {"flush_stride": -1}, {"flush_stride": 1.5}):
            args = dict(dt_fs=1.0, kT_eV=KT, logdir=d)
            args.update(kw)
            with T.assertRaises(ValueError):
                PlumedBias(None, ATOMS, ["input"] * 2, **args)
        assert not list(d.iterdir())


def test_invalid_positions_are_rejected_before_plumed():
    with tmpdir() as d:
        pb = one_walker(d)
        try:
            for x in (np.zeros((5, 3)), np.zeros((6, 2)), np.full((6, 3), np.nan)):
                with T.assertRaises(ValueError):
                    pb.bias(x)
            assert pb.step == 0
        finally:
            pb.finalize()


def test_flush_comment_and_endplumed_do_not_disable_flushing():
    for head, tail in (("# FLUSH mentioned in a comment\n", ""), ("", "ENDPLUMED\nFLUSH STRIDE=1000\n"),
                       ("", "e: ENDPLUMED\n")):
        with tmpdir() as d:
            pb = PlumedBias(None, [ATOMS[0]], [head + plumed_text(d, 0) + tail], 1.0, KT, d, flush_stride=PACE)
            try:
                pb.step = 1
                pb.bias(ATOMS[0].positions)
                pb.step = PACE
                pb.bias(ATOMS[0].positions)
                assert len(read_hills(d / "HILLS.0")) == 1, tail
            finally:
                pb.finalize()


# ---------------------------------------------------------------------------- restart (torch-sim >= 0.6)
class HarmonicToy(torch.nn.Module):
    """E = 0.5 k |x - x_ref|^2 per system; enough to move atoms around under the bias."""

    def __init__(self, ref, k=2.0):
        super().__init__()
        self.register_buffer("ref", torch.as_tensor(ref, dtype=torch.float32))
        self.k = k
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.compute_forces, self.compute_stress, self.memory_scales_with = True, False, "n_atoms"

    def forward(self, state, **kw):
        d = state.positions - self.ref
        e = torch.zeros(int(state.system_idx.max()) + 1, dtype=d.dtype)
        e.index_add_(0, state.system_idx, 0.5 * self.k * (d ** 2).sum(1))
        return {"energy": e, "forces": -self.k * d}


def make_pb(d, restart=None, pace=PACE):
    ref = np.concatenate([a.get_positions() for a in ATOMS])
    texts = [plumed_text(d, b, None if restart is None else restart[b], pace) for b in range(2)]
    return PlumedBias(HarmonicToy(ref), ATOMS, texts, 1.0, KT, d, flush_stride=pace)


def new_state(pb, seed=11):
    st = ts.io.atoms_to_state(ATOMS, device="cpu", dtype=torch.float32)
    st.rng = seed
    return nvt_langevin_init(st, pb, kT=KT)


DT, GAMMA = 1e-3 * U.time, 1.0 / (0.1 * U.time)        # 1 fs step, Langevin friction 1/(0.1 ps)


def advance(st, pb, a, b):
    for _ in range(a, b):
        st = nvt_langevin_step(st, pb, dt=DT, kT=KT, gamma=GAMMA)
    return st


def straight(d, n=80, ckpt_at=40, ckpt=None, pace=PACE):
    d.mkdir(exist_ok=True)
    pb = make_pb(d, pace=pace)
    st = new_state(pb)
    st = advance(st, pb, 0, ckpt_at)
    save_checkpoint(ckpt, st, pb)
    for b in range(2):
        shutil.copy(d / f"HILLS.{b}", d / f"HILLS.{b}.ckpt")   # on disk: deposition at step 40 and FLUSH at step 40
    x40 = st.positions.clone()
    st = advance(st, pb, ckpt_at, n)
    pb.finalize()
    return st.positions.clone(), x40


def resumed(d, src, ckpt, restore_rng=True, n=80, ckpt_at=40, pace=PACE):
    d.mkdir(exist_ok=True)
    files = []
    for b in range(2):
        shutil.copy(src / f"HILLS.{b}.ckpt", d / f"HILLS.{b}.restart")
        files.append(d / f"HILLS.{b}.restart")
    pb = make_pb(d, files, pace)
    st, info = load_checkpoint(ckpt, ATOMS)
    if not restore_rng:
        st.rng = 999
    resume_bias(pb, info)
    st = advance(st, pb, ckpt_at, n)
    pb.finalize()
    return st.positions.clone()


def test_resume_bias_refuses_a_used_wrapper_or_wrong_atoms():
    with tmpdir() as d:
        pb = one_walker(d)
        x = torch.as_tensor(ATOMS[0].positions)
        try:
            with T.assertRaises(ValueError):
                resume_bias(pb, {"bias_step": 5, "positions": x[:3]})
            assert pb.step == 0 and not pb._called
            pb.bias(x.numpy())                                    # a probe spends the first-call skip
            with T.assertRaisesRegex(ValueError, "not been called"):
                resume_bias(pb, {"bias_step": 5, "positions": x})
        finally:
            pb.finalize()


def test_checkpoint_roundtrip():
    with tmpdir() as d:
        pb = make_pb(d)
        st = advance(new_state(pb), pb, 0, 7)
        save_checkpoint(d / "ck.pt", st, pb, extra={"note": "x"})
        st2, info = load_checkpoint(d / "ck.pt", ATOMS)
        for k in ("positions", "momenta", "forces", "energy"):
            assert torch.equal(getattr(st, k), getattr(st2, k)), k
        assert torch.equal(st.rng.get_state(), st2.rng.get_state())
        assert info["bias_step"] == pb.step == 8 and info["extra"]["note"] == "x"
        pb.finalize()
        save_checkpoint(d / "ck.pt", st, pb)                     # a cleanly finalized bias can still be saved


class Patched(torch.nn.Module):
    """Wraps a model; edit(output dict) returns the entries to replace in the output."""

    def __init__(self, base, edit):
        super().__init__()
        self.base, self.edit = base, edit

    def forward(self, state, **kw):
        out = self.base(state, **kw)
        return {**out, **self.edit(out)}


def test_base_model_stress_is_not_passed_on():
    with tmpdir() as d:
        pb = make_pb(d)
        try:
            st = new_state(pb)
            pb.base = Patched(pb.base, lambda out: {"stress": torch.zeros(2, 3, 3)})
            assert "stress" not in pb(st)
        finally:
            pb.finalize()


def test_state_metadata_mutations_are_rejected_before_advancing_bias():
    with tmpdir() as d:
        pb = make_pb(d)
        try:
            for key in ("system_idx", "atomic_numbers", "masses", "pbc"):
                st = ts.io.atoms_to_state(ATOMS, device="cpu", dtype=torch.float32)
                pb(st)  # populate the metadata cache
                step = pb.step
                field = getattr(st, key)
                field.reshape(-1)[0] = True if key == "pbc" else field.reshape(-1)[0] + 1
                with T.assertRaisesRegex(ValueError, key):
                    pb(st)
                assert pb.step == step
            st = ts.io.atoms_to_state(ATOMS, device="cpu", dtype=torch.float32)
            st.positions = st.positions[:-1]
            with T.assertRaisesRegex(ValueError, "positions"):
                pb(st)
        finally:
            pb.finalize()


def test_checkpoint_rejects_different_systems_with_same_total_atom_count():
    with tmpdir() as d:
        pb = make_pb(d)
        try:
            save_checkpoint(d / "ck.pt", new_state(pb), pb)
            species = [a.copy() for a in ATOMS]
            species[0].numbers[0] = 8
            isotope = [a.copy() for a in ATOMS]
            isotope[0].set_masses([13.0] * 6)
            layout = [Atoms("C5"), Atoms("C7")]
            periodic = [a.copy() for a in ATOMS]
            for a in periodic:
                a.pbc = True
            cell = [a.copy() for a in ATOMS]
            cell[0].cell = np.eye(3) * 10
            for atoms, field in ((species, "atomic_numbers"), (isotope, "masses"),
                                 (layout, "system_idx"), (periodic, "pbc"), (cell, "cell")):
                with T.assertRaisesRegex(ValueError, field):
                    load_checkpoint(d / "ck.pt", atoms)
        finally:
            pb.finalize()


def test_checkpoint_rejects_unverifiable_legacy_format():
    with tmpdir() as d:
        path = d / "old.pt"
        torch.save({"format": 1}, path)
        with T.assertRaisesRegex(ValueError, "checkpoint format"):
            load_checkpoint(path, ATOMS)


def test_restart_matches_straight_run():
    with tmpdir() as d:
        x_end, x40 = straight(d / "a", ckpt=d / "ck.pt")
        y_end = resumed(d / "b", d / "a", d / "ck.pt")
        gap = float((x_end - y_end).abs().max())
        moved = float((x_end - x40).abs().max())
        print(f"  max |dx| restart vs straight after 40 more steps: {gap:.3e} A (positions moved by {moved:.3f} A)")
        assert gap < 1e-5, gap
        ha, hb = read_hills(d / "a" / "HILLS.0"), read_hills(d / "b" / "HILLS.0.restart")
        assert len(ha) == len(hb) == 4, (len(ha), len(hb))              # deposits at 20, 40, 60, 80
        assert np.abs(ha - hb).max() < 1e-6, np.abs(ha - hb).max()


def test_restart_without_rng_diverges():
    with tmpdir() as d:
        x_end, _ = straight(d / "a", ckpt=d / "ck.pt")
        y_end = resumed(d / "b", d / "a", d / "ck.pt", restore_rng=False)
        gap = float((x_end - y_end).abs().max())
        print(f"  max |dx| without rng restore: {gap:.3e} A")
        assert gap > 1e-3, gap


def test_restart_in_new_process_matches_straight_run():
    with tmpdir() as d:
        x_end, _ = straight(d / "a", ckpt=d / "ck.pt")
        script = ("import runpy, numpy as np; from pathlib import Path; "
                  "m = runpy.run_path(__import__('sys').argv[1]); d = Path(__import__('sys').argv[2]); "
                  "x = m['resumed'](d / 'b', d / 'a', d / 'ck.pt'); "
                  "np.save(d / 'resumed.npy', x.numpy())")
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(d)],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert np.max(np.abs(x_end.numpy() - np.load(d / "resumed.npy"))) < 1e-5


def test_restart_when_next_step_deposits():
    """The first resumed step is a deposition step (PACE=20 saved at 39, PACE=1): no hill may go missing."""
    for pace, ck in ((20, 39), (1, 40)):
        with tmpdir() as d:
            x_end, _ = straight(d / "a", n=ck + 25, ckpt_at=ck, ckpt=d / "ck.pt", pace=pace)
            y_end = resumed(d / "b", d / "a", d / "ck.pt", n=ck + 25, ckpt_at=ck, pace=pace)
            ha, hb = read_hills(d / "a" / "HILLS.0"), read_hills(d / "b" / "HILLS.0.restart")
            assert len(ha) == len(hb) and np.abs(ha - hb).max() < 1e-6, (pace, ck, len(ha), len(hb))
            assert float((x_end - y_end).abs().max()) < 1e-5, (pace, ck)


def test_malformed_model_output_and_checkpoint_are_rejected():
    with tmpdir() as d:
        pb = make_pb(d)
        try:
            st = advance(new_state(pb), pb, 0, 3)
            base = pb.base
            # energy [2, 1] would broadcast to [2, 2]
            pb.base = Patched(base, lambda out: {"energy": out["energy"].unsqueeze(-1)})
            with T.assertRaisesRegex(ValueError, "energy"):
                pb(st)
            pb.base, pb.failed = base, False                          # a direct probe call left the state unchanged
            save_checkpoint(d / "ck.pt", st, pb)
            with T.assertRaises(Exception):
                save_checkpoint(d / "ck.pt", st, pb, extra={"path": Path("x")})
            assert not (d / "ck.pt.tmp").exists()                      # the failed write left no temporary file
            load_checkpoint(d / "ck.pt", ATOMS)                       # the good checkpoint is still in place
            with T.assertRaisesRegex(ValueError, "float32"):
                load_checkpoint(d / "ck.pt", ATOMS, dtype=torch.float64)
            ck = torch.load(d / "ck.pt", weights_only=True)
            for k, bad in (("forces", ck["forces"][:1]), ("positions", ck["positions"] * np.nan)):
                torch.save({**ck, k: bad}, d / "bad.pt")
                with T.assertRaisesRegex(ValueError, k):
                    load_checkpoint(d / "bad.pt", ATOMS)
            torch.save({**ck, "identity": {}}, d / "bad.pt")
            with T.assertRaises(KeyError):
                load_checkpoint(d / "bad.pt", ATOMS)
        finally:
            pb.finalize()



def test_non_finite_state_and_outputs_are_refused():
    with tmpdir() as d:
        pb = make_pb(d)
        try:
            st = advance(new_state(pb), pb, 0, 3)
            save_checkpoint(d / "ck.pt", st, pb)
            good = (d / "ck.pt").read_bytes()
            st.forces = st.forces * np.nan
            with T.assertRaisesRegex(ValueError, "forces"):
                save_checkpoint(d / "ck.pt", st, pb)
            assert (d / "ck.pt").read_bytes() == good and not (d / "ck.pt.tmp").exists()
            base, step = pb.base, pb.step
            pb.base = Patched(base, lambda out: {"forces": out["forces"] * np.nan})
            with T.assertRaises(FloatingPointError):
                pb(st)
            assert pb.step == step and pb._plumed                      # not advanced, still usable
            with T.assertRaisesRegex(ValueError, "mid-step"):
                save_checkpoint(d / "ck.pt", st, pb)                    # the state of a failed step is not saved
            pb.base = base
            x = np.concatenate([a.positions for a in ATOMS])
            x[1] = x[0]                                                 # DISTANCE between overlapping atoms
            with T.assertRaises(FloatingPointError):
                pb.bias(x)
            assert not pb._plumed                                       # finalized
            assert (d / "ck.pt").read_bytes() == good
        finally:
            pb.finalize()


def test_state_extras_survive_restart():
    with tmpdir() as d:
        pb = make_pb(d)
        try:
            st = advance(new_state(pb), pb, 0, 3)
            if not hasattr(st, "_system_extras"):
                print("  torch-sim without state extras: nothing to check")
                return
            st._system_extras["charge"] = torch.tensor([0.0, -1.0])
            st._atom_extras["tag"] = torch.arange(12.0)
            save_checkpoint(d / "ck.pt", st, pb)
            st2, _ = load_checkpoint(d / "ck.pt", ATOMS)
            assert torch.equal(st2.system_extras["charge"], st.system_extras["charge"])
            assert torch.equal(st2.atom_extras["tag"], st.atom_extras["tag"])
            st._system_extras["charge"] = torch.tensor([0.0, np.nan])
            with T.assertRaisesRegex(ValueError, "charge"):
                save_checkpoint(d / "ck.pt", st, pb)
        finally:
            pb.finalize()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
