# Scale-shape dual Newton — Section 7 reproduction code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19889724.svg)](https://doi.org/10.5281/zenodo.19889724)

Companion code to Section 7 of

> N. Barnfield, J. V. Burke, M. P. Friedlander, T. Hoheisel,
> *A scale-shape dual Newton method for entropic least squares*, 2026.
> arXiv: [`XXXX.XXXXX`](https://arxiv.org/abs/XXXX.XXXXX)  ·  DOI: `10.XXXX/XXXXXX`

This repository reproduces the three numerical figures of Section 7 on the
uniform-electron-gas (UEG) analytic-continuation test problem of Chuna et al.
(2025). The solver is a damped inexact Newton method on the scale-shape dual,
applied to the optimality system F(y, τ) = 0 of an entropy-regularized
least-squares problem; see the paper for the algorithm and analysis.

## Quick reproduction

```bash
git clone https://github.com/MPF-Optimization-Laboratory/2026-Barnfield-Burke-Friedlander-Hoheisel.git
cd 2026-Barnfield-Burke-Friedlander-Hoheisel
uv sync          # install pinned environment (Python 3.11)
make figures     # regenerate all figures into figures/
```

Total wall-clock time: ~3 minutes on a 2023 laptop. No GPU required.

## Figure map

Figure 7.2 in the paper is a 2×3 composite: the top row studies overflow
resilience (Z ∈ {2⁴, 2⁶, 2⁸, 2¹⁰}) and the bottom row studies scale recovery
(Z ∈ {1, 10, 100}). Its six panels and two per-row legend strips are emitted
as separate PDFs and assembled in the LaTeX source via `subcaption`.

| Paper          | Script                                    | Output                                        | Wall-clock |
|----------------|-------------------------------------------|-----------------------------------------------|------------|
| Figure 7.1     | `scripts/fig_7_1_problem_data.py`         | `figures/problem_data.pdf`                    | ~10 s      |
| Figure 7.2     | `scripts/fig_7_iteration_trajectories.py` | `figures/trajectory_{a..f}.pdf`, `figures/trajectory_legend_row{1,2}.pdf` | ~45 s      |
| Figure 7.3     | `scripts/fig_7_4_regularization_path.py`  | `figures/regularization_path_rel.pdf`         | ~2 min     |

## Requirements

- Python 3.11 or 3.12 (pinned via `requires-python`)
- [uv](https://docs.astral.sh/uv/) for environment management (recommended), or
  any tool that can install from the exported `requirements.txt`
- A LaTeX installation with Computer Modern fonts and `amsmath`, for
  `matplotlib` text rendering (`text.usetex = True`). On Debian/Ubuntu:
  `apt-get install texlive-latex-base texlive-latex-recommended
  texlive-fonts-recommended dvipng cm-super`. On macOS: MacTeX or BasicTeX.
  To regenerate figures **without** a LaTeX installation, comment out the
  `text.usetex` line in `src/scaleshape/figstyle.py`.
- ~2 GB RAM; runs single-threaded

## Installation (alternatives)

Without `uv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make figures RUN=python
```

## Running individual experiments

```bash
uv run python scripts/fig_7_1_problem_data.py
uv run python scripts/fig_7_iteration_trajectories.py
uv run python scripts/fig_7_4_regularization_path.py
```

To generate the UEG test problem at a custom scale and save it to
`data/ueg_problem.npz`:

```bash
uv run python -m scaleshape.ueg --Z 5.0 --noise 1e-3 --seed 123
```

## Repository layout

```
.
├── src/scaleshape/     package: solver, UEG problem generator, utilities
│   ├── solver.py        core algorithm (scaleshape_solve, Model)
│   ├── ueg.py           generate_ueg_problem + CLI
│   ├── utils.py         naive_dual_newton, lambert_w_bounds
│   └── figstyle.py      matplotlib style
├── scripts/             one script per Section 7 figure
├── data/                synthetic-UEG_testproblem.npz (committed; 819 KB)
├── figures/             generated figures (committed; regenerable via `make figures`)
├── tests/test_smoke.py  fast correctness tests (<60 s)
├── pyproject.toml       project metadata and dependencies
├── uv.lock              transitively pinned resolution
├── requirements.txt     format-stable fallback
├── Makefile             reproduction entry point
├── CITATION.cff         citation metadata
├── .zenodo.json         Zenodo release metadata
└── LICENSE              MIT
```

## Data provenance

`data/synthetic-UEG_testproblem.npz` (819 KB) contains the physics inputs for
the UEG analytic-continuation experiment:

| key         | description                                                      |
|-------------|------------------------------------------------------------------|
| `x`         | Completed Mermin DSF at k ≈ 0.30 k<sub>D,e</sub> (ground truth; sum ≈ 1) |
| `mu`        | RPA-model DSF (prior; sum ≈ 1)                                    |
| `taus`      | Imaginary-time grid (m = 201 points)                              |
| `omegas`    | Frequency grid (n = 500 points)                                   |

The kernel A is reconstructed on the fly from `taus` and `omegas` (see
`build_kernel` in `src/scaleshape/ueg.py`), which avoids carrying the
201×500 matrix in the base data file.

The base data was produced by T. Chuna. For physics background and the
kernel derivation, see

> T. Chuna, N. Barnfield, T. Dornheim, M. P. Friedlander, T. Hoheisel,
> *Dual formulation of the maximum entropy method applied to analytic
> continuation of quantum Monte Carlo data*, 2025.

## Reproducibility notes

- All random seeds are fixed (`seed = 42` for the base, `seed = 42 + Z` for
  per-Z instances). Seeds are visible at the top of each script.
- The RPA prior has exponentially decaying tails; 251 of 500 entries lie
  below 10⁻¹⁰. The solver requires μᵢ > 0, so every script floors
  `mu ← max(mu, 1e-16)` and renormalizes. On the effective support
  (entries above 10⁻¹⁰) the relative shift is bounded by 10⁻⁶; the floor
  is inconsequential.
- Exact numerical output may drift across BLAS/matplotlib versions, but
  figure structure (convergence shape, τ recovery, residual decay) is
  stable to within the solver tolerance (10⁻⁸ to 10⁻¹⁰).
- CI (`.github/workflows/ci.yml`) runs the smoke test on every push and
  regenerates Figure 7.1 as an artifact.

## Archival references

- **Zenodo DOI**: [`10.5281/zenodo.19889724`](https://doi.org/10.5281/zenodo.19889724)
- **Software Heritage ID**: `swh:1:dir:XXXX...`                *(added after public release)*
- **GitHub source**: <https://github.com/MPF-Optimization-Laboratory/2026-Barnfield-Burke-Friedlander-Hoheisel>

## Citation

Please cite both the paper and this software. See `CITATION.cff` (rendered by
GitHub with a "Cite this repository" button). In BibTeX:

```bibtex
@article{BarnfieldBurkeFriedlanderHoheisel2026,
  author  = {Barnfield, Nicholas and Burke, James V. and Friedlander, Michael P. and Hoheisel, Tim},
  title   = {A scale-shape dual {N}ewton method for entropic least squares},
  year    = {2026},
}

@software{BarnfieldBurkeFriedlanderHoheisel2026Code,
  author  = {Barnfield, Nicholas and Burke, James V. and Friedlander, Michael P. and Hoheisel, Tim},
  title   = {A scale-shape dual {N}ewton method for entropic least squares ({S}ection 7 reproduction code)},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.19889724},
  url     = {https://github.com/MPF-Optimization-Laboratory/2026-Barnfield-Burke-Friedlander-Hoheisel},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
