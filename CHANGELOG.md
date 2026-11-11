# Changes

## Unreleased

- Positions are moved to the CPU before the float64 conversion, so a model on MPS works.
- An error raised by PLUMED in a direct `bias()` call sets `failed`, as it already did through the model call.
- `save_checkpoint` syncs the temporary file to disk before it replaces the previous checkpoint.
- State extras are read and written through torch-sim's public `system_extras` / `atom_extras`.
- New tests: OPES_METAD and OPES_METAD_EXPLORE restart (needs the opes module), PLUMED error in a direct `bias()` call.

## 0.1.0 (2026-11-10)

Initial release.

- `PlumedBias`: adds the bias of one PLUMED instance per system to a torch-sim model (non-periodic systems, eV/Angstrom/fs).
- `save_checkpoint`, `load_checkpoint` and `resume_bias` for restarting unconstrained NVE and Langevin NVT runs together with a PLUMED HILLS or STATE snapshot.
- Examples (CPU restart, SevenNet), an overhead benchmark, an n-butane CUDA OPES restart harness and a Linux CPU CI workflow.
