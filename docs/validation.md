# Validation

The packaged tests use a harmonic model and run on CPU. They check the force coupling, the METAD restart, FLUSH handling, input and checkpoint validation, and failure handling.

```bash
pytest
# Without pytest:
python tests/test_plumed_bias.py
```

## Recorded runs

Version 0.1.0:

| Environment | Result |
|---|---|
| macOS arm64 CPU: Python 3.12.16, torch 2.15.0, torch-sim 0.6.2, ASE 3.29.0, PLUMED kernel 2.10.2, binding 2.9.2 | All tests pass, also from the installed wheel outside the source tree. Minimal example: restart difference 0.0 Angstrom |
| GitHub Actions ubuntu-24.04 CPU: Python 3.12, torch 2.12.1+cpu, torch-sim 0.6.0, PLUMED and binding 2.9.2 | Same, from the installed wheel. The extras test is skipped on torch-sim 0.6.0 |

Before the 0.1.0 fixes, an earlier revision also ran on Linux CPU with torch 2.12.1 (CPU and CUDA builds), torch-sim 0.6.0 and PLUMED 2.9.2. There the SevenNet 0.13.0 example ran 40 steps for two water replicas and deposited two METAD hills per system.

## n-Butane CUDA / OPES restart (2026-11-07)

SevenNet-omni (`omol25_low`, SevenNet 0.13.0) on an NVIDIA RTX PRO 5000 Blackwell, torch 2.12.1+cu130, torch-sim 0.6.0 and PLUMED 2.9.2. The acceptance limits were fixed in `butane_validation.json` before the run and all were met. The case is non-periodic n-butane (14 atoms), carbon-chain torsion, 300 K, 0.5 fs and OPES_METAD_EXPLORE. GPU elapsed time was 1 h 29 min 17 s.

A 20 ps straight run saved MD state/RNG and an OPES STATE snapshot at 10 ps. Separate processes ran the remaining 10 ps, a 20 ps repeated straight run, a 1 ps restart with a different RNG, and a 1 ps two-system batch.

| Check | Result |
|---|---|
| Loaded checkpoint tensors and RNG | Exact matches before continuing |
| First resumed step: max position difference | 0.0 Angstrom |
| First resumed step: max force difference | 7.75e-7 eV/Angstrom (limit 1e-4) |
| First resumed step: bias difference | 1.21e-17 eV (limit 1e-5) |
| OPES update schedule after checkpoint | Same 200 update times |
| Position difference 1 ps after restart | 4.48e-5 Angstrom; without restored RNG: 2.73 Angstrom |
| Changing only the other system's coordinates | First system's bias/force difference 0.0 |

End-point position differences were 0.0201 Angstrom for restart versus straight (10 ps after the checkpoint) and 2.274 Angstrom for repeat versus straight (20 ps). No acceptance limit was set for these.

[Case and results](../benchmarks/results/butane_opes_restart.json) and [reproduction harness](../benchmarks/validate_butane.py) are included. Harness requirements are listed in [usage](usage.md#recorded-opes-restart-case). The run used the packaged harness with an earlier `resume_bias`, which did not replay the checkpoint step, so the first resumed step missed one OPES variance update. It has not been re-run with 0.1.0.

## What each test checks

| test | checks |
|---|---|
| `test_units_restraint` | a RESTRAINT in eV/Angstrom^2 gives 0.5 k (d - d0)^2 and the matching force, to 1e-9 |
| `test_force_matches_finite_difference` | bias force equals the central difference of the bias energy (METAD with deposited hills) |
| `test_metad_matches_numpy_reference_and_reload` | METAD bias equals a numpy recomputation from the HILLS file; a fresh instance restarted from a copy gives the same value |
| `test_systems_are_independent` | the bias of one system does not depend on the positions of another |
| `test_resume_bias_refuses_a_used_wrapper_or_wrong_atoms` | `resume_bias` refuses checkpoint positions of the wrong shape without changing the step, and refuses a wrapper whose `bias` was already called |
| `test_checkpoint_roundtrip` | tensors and random generator state survive save and load |
| `test_restart_matches_straight_run` | 80 steps straight equal 40 steps, checkpoint, new objects with PLUMED RESTART, 40 more steps |
| `test_restart_without_rng_diverges` | the same restart without the generator state does not follow the straight run |
| `test_hills_on_disk_without_flush` | prints whether a deposited hill is on disk without FLUSH, and checks that it is with `flush_stride` |
| `test_invalid_constructor_inputs_are_rejected_before_creating_files` | rejects mismatched input counts, empty/periodic systems and invalid parameters |
| `test_invalid_positions_are_rejected_before_plumed` | rejects wrong coordinate shapes and non-finite positions |
| `test_flush_comment_and_endplumed_do_not_disable_flushing` | a comment containing FLUSH does not suppress the actual action; with ENDPLUMED, the FLUSH line goes before it and a FLUSH after it is ignored |
| `test_state_metadata_mutations_are_rejected_before_advancing_bias` | rejects changed system membership, species, masses or periodicity, including in-place edits |
| `test_checkpoint_rejects_different_systems_with_same_total_atom_count` | refuses species, isotope, partition, periodicity and cell mismatches |
| `test_checkpoint_rejects_unverifiable_legacy_format` | refuses checkpoints without identity information |
| `test_restart_in_new_process_matches_straight_run` | restores the on-disk checkpoint and HILLS in a separate Python process |
| `test_restart_when_next_step_deposits` | PACE=20 saved at step 39 and PACE=1 saved at step 40: hill count, hill values and final positions match the uninterrupted run |
| `test_malformed_model_output_and_checkpoint_are_rejected` | bad model output shapes, unloadable `extra`, a different dtype and corrupted checkpoints are refused; the previous checkpoint survives a failed save |
| `test_base_model_stress_is_not_passed_on` | a `stress` returned by the base model is not in the wrapper output |
| `test_non_finite_state_and_outputs_are_refused` | NaN states are not saved; NaN model output raises without advancing the step and blocks the next save; a NaN PLUMED bias finalizes the wrapper |
| `test_state_extras_survive_restart` | system and atom extras of the torch-sim state survive save and load, and a NaN extra is not saved (skipped on torch-sim without extras) |

The restart tests need torch-sim >= 0.6. On GPU, repeated runs diverge numerically (2.274 Angstrom between two straight 20 ps runs, above); the CPU restart difference of 0.0 does not apply there.

## Not tested

- Free energies, rates, model accuracy, sampling efficiency or convergence.
- Full MD restart with OPES_METAD (non-EXPLORE), and with constrained states.
- A bias shared between batch systems.
- Model and version combinations other than those listed above.

End-to-end restart has been run for METAD on CPU and for OPES_METAD_EXPLORE on CUDA.

## PLUMED details

- The HILLS `height` column of a well-tempered METAD is the deposited height times gamma/(gamma-1); the bias felt by the system is height times (gamma-1)/gamma. A recomputation from the file has to include that factor.
- PLUMED's stretched Gaussian is cut where 0.5 * sum((s - c)/sigma)^2 reaches 6.25 and rescaled by 1/(1 - exp(-6.25)).
- `setMDEnergyUnits` takes PLUMED's own value of 1 eV in kJ/mol, 96.4853075. The CODATA 2018 value 96.48533212 gives bias energies that differ by 2.6e-7 (relative; visible on large wall energies).
- METAD and OPES skip the history update on the first call an instance sees, whatever the step number (`isFirstStep_` in PLUMED 2.10 `MetaD.cpp` and `OPESmetad.cpp`).
- METAD `ADAPTIVE=DIFF` (see [usage](usage.md#restart-two-parts)): CPU test, first post-restart hill sigma 0.223 vs 0.372; positions differ by 0.003 Angstrom after 60 steps.
- OPES has no file-based walker sharing, only `WALKERS_MPI`. File-sharing METAD walkers: see [platform notes](usage.md#platform-notes).
- OPES_METAD_EXPLORE normalises the kernel sum by the sample counter, OPES_METAD by the sum of weights.

