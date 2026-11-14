# plumed-bias — Metadynamics and OPES for torch-sim

[![CI](https://github.com/Chegal-Jihwan/plumed-bias/actions/workflows/ci.yml/badge.svg)](https://github.com/Chegal-Jihwan/plumed-bias/actions/workflows/ci.yml)

plumed-bias connects [PLUMED](https://www.plumed.org/) to [torch-sim](https://github.com/TorchSim/torch-sim) for enhanced-sampling molecular dynamics. It runs batched simulations of independent, non-periodic systems, each with its own PLUMED bias instance and history.

Metadynamics (MetaD), well-tempered metadynamics and OPES are available through the PLUMED kernel you install; OPES needs a kernel built with the opes module. `PlumedBias` adds PLUMED's bias energy and forces to the output of the base model, which can be a machine-learning interatomic potential, and the torch-sim integrator stays in charge of the MD.

Version 0.1.0 (2026-11-10). Tested on macOS arm64 and Linux CPU; the versions are listed in [validation](docs/validation.md#recorded-runs). plumed-bias is a separate project, not part of PLUMED or torch-sim.

## Key features

- Independent PLUMED biases across the systems of a batch
- Works with any torch-sim model, including machine-learning interatomic potentials
- MD checkpoint and restart, paired with PLUMED HILLS or OPES STATE snapshots
- NVE and Langevin NVT integration
- CPU examples, and a recorded CUDA validation on one GPU setup

## Scope and limitations

- Non-periodic systems with a fixed atom layout.
- Checkpoint and restart of unconstrained NVE and Langevin NVT runs.
- Not supported: periodic boundaries, bias stress, NPT, a bias shared between batch systems.
- GPU runs are recorded for one CUDA setup only (see [validation](docs/validation.md#n-butane-cuda--opes-restart-2026-11-07)).

## Quick start

```bash
git clone https://github.com/Chegal-Jihwan/plumed-bias.git
cd plumed-bias
conda create -n plumed-bias -c conda-forge python=3.12 plumed py-plumed
conda activate plumed-bias
python -m pip install torch
python -m pip install -e ".[test]"
export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"   # macOS: libplumedKernel.dylib
python examples/minimal_example.py
```

The example runs two systems with a small harmonic model, saves at step 200, restarts and continues to step 400. It then prints the largest position difference from an uninterrupted run:

```text
max |difference| (A): 0.0
```

## Installation

PLUMED and its Python binding come from conda-forge, as in the quick start. Install torch before the package: the CPU build, or the CUDA build for your system from [pytorch.org](https://pytorch.org/). Then run the tests from the source checkout:

```bash
pytest
```

Python 3.12 or newer is required. `PLUMED_KERNEL` must point to the kernel library of the intended PLUMED installation. On macOS, read the [platform notes](docs/usage.md#platform-notes) about the OpenMP runtime first. For OPES, use a PLUMED kernel built with the opes module; the conda-forge 2.9.2 kernel does not include it.

## Usage

With a base model, a list of ASE Atoms and one PLUMED input per system:

```python
from plumed_bias import PlumedBias

pb = PlumedBias(base_model, atoms_list, inputs,
                dt_fs=1.0, kT_eV=kT, logdir=out, flush_stride=500)
try:
    state = nvt_langevin_init(state, pb, kT=kT)  # PLUMED step 0
    state = nvt_langevin_step(state, pb, dt=dt, kT=kT, gamma=gamma)
finally:
    pb.finalize()
```

PLUMED inputs need `UNITS LENGTH=A ENERGY=eV` and `NOPBC`. Each wrapper call advances the PLUMED step, so the integrator must evaluate the model once per MD step. The [usage guide](docs/usage.md) covers the model contract and restarts.

Restarting needs the MD checkpoint (`save_checkpoint`, `load_checkpoint`, `resume_bias`) and a copy of the PLUMED HILLS or STATE file taken at the same step. The [restart section](docs/usage.md#restart-two-parts) gives the sequence.

## Examples

- [Minimal example](examples/minimal_example.py): CPU run with a checkpoint and restart.
- [SevenNet example](examples/sevennet_md.py): metadynamics with a machine-learning interatomic potential, with a [walkthrough](docs/usage.md#sevennet-example).

## Validation and benchmarks

- [Validation](docs/validation.md): what the tests check, recorded environments, and a CUDA OPES restart of n-butane run with an earlier `resume_bias`.
- [Overhead benchmark](benchmarks/README.md): cost of the bridge itself, with a recorded macOS CPU run.

## Development

See [CONTRIBUTING](CONTRIBUTING.md) and the [changelog](CHANGELOG.md).

## Citation

Cite the software and methods your run uses. Entries are also in [CITATION.cff](CITATION.cff).

Software:

- PLUMED: G. A. Tribello, M. Bonomi, D. Branduardi, C. Camilloni, G. Bussi, Comput. Phys. Commun. 185, 604 (2014), [doi:10.1016/j.cpc.2013.09.018](https://doi.org/10.1016/j.cpc.2013.09.018); The PLUMED consortium, Nat. Methods 16, 670 (2019), [doi:10.1038/s41592-019-0506-8](https://doi.org/10.1038/s41592-019-0506-8)
- torch-sim: O. Cohen et al., AI for Science 1, 025003 (2025), [doi:10.1088/3050-287X/ae1799](https://doi.org/10.1088/3050-287X/ae1799)
- ASE: A. Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017), [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)
- SevenNet (examples and CUDA check): Y. Park, J. Kim, S. Hwang, S. Han, J. Chem. Theory Comput. 20, 4857 (2024), [doi:10.1021/acs.jctc.4c00190](https://doi.org/10.1021/acs.jctc.4c00190); SevenNet-Omni: J. Kim et al., Nat. Commun. 17, 3432 (2026), [doi:10.1038/s41467-026-70195-8](https://doi.org/10.1038/s41467-026-70195-8)

Methods used in the examples and checks:

- Metadynamics: A. Laio, M. Parrinello, Proc. Natl. Acad. Sci. USA 99, 12562 (2002), [doi:10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399)
- Well-tempered metadynamics: A. Barducci, G. Bussi, M. Parrinello, Phys. Rev. Lett. 100, 020603 (2008), [doi:10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603)
- OPES: M. Invernizzi, M. Parrinello, J. Phys. Chem. Lett. 11, 2731 (2020), [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497)
- OPES explore: M. Invernizzi, M. Parrinello, J. Chem. Theory Comput. 18, 3988 (2022), [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152)

## License and dependencies

MIT, see [LICENSE](LICENSE). Dependencies (numpy, torch, ASE, torch-sim, PLUMED) and model checkpoints keep their own licenses and are not bundled. The optional SevenNet example requires SevenNet and its dependencies.

## Author

Jihwan Chegal (wlghks1294@khu.ac.kr), Computational & Data-driven Chemistry Lab, Kyung Hee University.
