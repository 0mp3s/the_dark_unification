# How to Run the Code

## Quick Start

```bash
git clone https://github.com/0mp3s/the_dark_unification
cd the_dark_unification
pip install -r requirements.txt
```

All observational data is already embedded in `data/observational_database.json`.  
No external downloads required.

---

## Chapter 1 — SIDM Pipeline

### Execution Order (full pipeline)

```
1. v22_raw_scan_fast.py     ← grid scan, produces parameter space
2. smart_scan.py            ← refines viable region
3. boltzmann_correction.py  ← adds relic density constraint
4. opusB_run_mcmc.py        ← MCMC posterior (slow, ~9h CPU)
5. plot_island.py           ← figures: island of viability, corner plot
```

### Script Reference

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `v22_raw_scan_fast.py` | `data/observational_database.json` (via core) | `data/benchmark_points.csv` | Grid scan over (mχ, mφ, α). JIT-compiled. ~20 min |
| `smart_scan.py` | `data/benchmark_points.csv` | `data/all_viable_representative.csv` | Refines viable region with finer grid |
| `boltzmann_correction.py` | `data/all_viable_representative.csv` | `data/sidm_relic_viable_points.csv` | Adds Ωh²=0.120 constraint |
| `opusB_run_mcmc.py` | `data/sidm_relic_viable_points.csv` | `data/mcmc_checkpoint.npy` | 32 walkers × 5000 steps, emcee |
| `plot_island.py` | `data/all_viable_representative.csv` | `paper/figures/island.png` | Island of viability plot |
| `benchmark_extractor.py` | `data/all_viable_representative.csv` | stdout | Prints BP1, BP9, BP16, MAP values |
| `gelman_rubin.py` | `data/mcmc_checkpoint.npy` | stdout | Convergence diagnostics (R̂ < 1.01) |
| `born_validation.py` | none (analytic) | stdout | Confirms Born fails 88–135× vs VPM |
| `partial_wave_decomposition.py` | none (analytic) | stdout | Shows ℓ convergence |
| `vpm_born_ratio.py` | `data/observational_database.json` | `paper/figures/vpm_born.png` | Fig: ratio σ_VPM / σ_Born vs velocity |
| `velocity_slope.py` | `data/observational_database.json` | `paper/figures/velocity_slope.png` | σ_T(v) across 13 systems |
| `error_budget.py` | `data/mcmc_checkpoint.npy` | stdout | Uncertainty budget per observable |
| `mediator_cosmology.py` | `data/observational_database.json` | stdout | ΔN_eff, BBN, CMB constraints |
| `fermi_lat_dsph.py` | `data/observational_database.json` | stdout | Indirect detection bounds |
| `pipeline.py` | `data/sidm_relic_viable_points.csv` | `data/chi2_fit_results.csv` | Full χ² fit to 13 systems |
| `run_mcmc.py` | `data/sidm_relic_viable_points.csv` | `data/mcmc_checkpoint.npy` | Alternative MCMC runner (diagnostic) |
| `autocorr_diagnostic.py` | `data/mcmc_checkpoint.npy` | stdout + plot | Autocorrelation time τ |

### How Imports Work

All scripts import observational data through the config loader:

```python
from core.global_config import GC

obs    = GC.observations()           # 13 observational systems
fornax = GC.fornax_halo()            # Read+2019 halo parameters
walker = GC.walker2009_fornax()      # Walker+2009 σ_los profile
cosmo  = GC.cosmological_constants() # Planck 2018
bp     = GC.benchmark("MAP")         # MAP benchmark point dict
path   = GC.csv_path("relic_viable") # Path to sidm_relic_viable_points.csv
```

Run scripts from the `chapter1_sidm/` directory:
```bash
cd chapter1_sidm
python opusB_run_mcmc.py
```

---

## Chapter 2 — Path Integral Tests

These scripts verify the 7-layer derivation. They are self-contained — no CSV input needed.

| Script | What it verifies | Runtime |
|--------|-----------------|---------|
| `lagrangian_path_integral.py` | Full 7-layer chain from Z[J] to Ωh² | ~1 min |
| `test_PI8_standalone_colab.py` | Layer 5: Matsubara finite-T potential | <1 min |
| `test_PI8_phase_transition_relic.py` | Layer 5: vacuum stability at freeze-out | <1 min |
| `test_PI9_pwave_standalone_colab.py` | p-wave suppression check | <1 min |
| `test_PI11_deriv_coupling_standalone_colab.py` | Derivative coupling consistency | <1 min |
| `test_PI12_clockwork_standalone_colab.py` | Layer 6: clockwork / instanton | <1 min |
| `test_PI12_v2_proper_KT_colab.py` | Layer 6: KT-formalism instanton | <1 min |
| `test_PI13_sommerfeld_standalone_colab.py` | Sommerfeld enhancement vs VPM | <1 min |
| `test_PI14_17_combined_standalone_colab.py` | Layers 4+5: CW + Matsubara combined | <1 min |
| `test_PI14_17_v2_proper_KT_colab.py` | Layers 4+5: KT formalism | <1 min |
| `test_PI18_19_boltzmann_hubble_colab.py` | Layer DE: H₀ = 67.4 prediction | <1 min |
| `test_physics_checks.py` | All layers simultaneously | ~2 min |

Run:
```bash
cd chapter2_path_integral
python lagrangian_path_integral.py   # full chain
python test_physics_checks.py        # all unit tests
```

---

## Chapter 3 — Dark Energy / T-Breaking

Self-contained scripts, no CSV input. Each tests one aspect of the dark EM duality.

| Script | What it computes | Key output |
|--------|-----------------|------------|
| `a4_dark_sector_model.py` | A₄ symmetry group, CGC tables | θ_relic = 19.47° |
| `dark_axion_full.py` | Full dark axion potential V_bare(σ) | f, m_σ, w_DE |
| `dark_qcd_consistency.py` | Dark QCD confinement, ΔN_eff | Λ_d ~ 2.05 meV |
| `qcd_scale_coincidence.py` | Why Λ_d ~ √(H₀ M_Pl) | numerical coincidence table |
| `theta_topological.py` | Topological charge, instanton suppression | S_E ~ 10¹²¹ |
| `sigma_mass_protection.py` | Why m_σ is technically natural | radiative stability |
| `sigma_radiative_stability.py` | Loop corrections to m_σ | Δm_σ/m_σ < 1% |
| `fifth_force_constraints.py` | β = M_Pl/f, chameleon screening | β ≈ 5.0 |
| `freeze_out_analysis_corrected.py` | Freeze-out temperature, Ωh² for dark axion | T_fo, Ω_σ = 0.69 |
| `run_pipeline.py` | All Ch.3 checks sequentially | Combined report |

Run:
```bash
cd chapter3_t_breaking
python run_pipeline.py   # runs all checks in order
```

---

## Reproducing the Paper Figures

After running the full pipeline, figures are written to `paper/figures/`.

| Figure | Produced by | Data needed |
|--------|------------|-------------|
| Island of viability | `chapter1_sidm/plot_island.py` | `data/all_viable_representative.csv` |
| MCMC corner plot | `chapter1_sidm/autocorr_diagnostic.py` | `data/mcmc_checkpoint.npy` |
| σ_T(v) vs 13 systems | `chapter1_sidm/velocity_slope.py` | `data/observational_database.json` |
| VPM vs Born ratio | `chapter1_sidm/vpm_born_ratio.py` | `data/observational_database.json` |
| H₀ chain diagram | `chapter2_path_integral/test_PI18_19_boltzmann_hubble_colab.py` | none |

---

## Compiling the Paper PDF

MiKTeX or TeX Live required.

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # run twice for references
```

Output: `paper/main.pdf`

---

## Parameter Space Summary

The viable parameter island (from `data/all_viable_representative.csv`):

| Parameter | Range |
|-----------|-------|
| m_χ (DM mass) | 10–100 GeV |
| m_φ (mediator mass) | 7.6–14.8 MeV |
| α (coupling) | 2×10⁻⁴ – 4×10⁻³ |

MAP point: `m_χ = 98.2 GeV`, `m_φ = 9.66 MeV`, `α = 3.27×10⁻³`
