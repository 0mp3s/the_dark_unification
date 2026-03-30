# Data Sources and Observational Datasets

All observational data used in this paper is embedded directly in
`data/observational_database.json`. No external files need to be downloaded.
Below is a full accounting of every dataset, its source, and where it appears.

---

## 1. Observational Constraints on σ_T/m (13 systems)

**File:** `data/observational_database.json` → key `"observations"`

| System | v [km/s] | σ/m [cm²/g] | Reference |
|--------|----------|-------------|-----------|
| Draco dSph | 12 | 0.1–2.0 | Kaplinghat, Tulin & Yu 2016 |
| Fornax dSph | 12 | 0.2–3.0 | Kaplinghat, Tulin & Yu 2016 |
| NGC 2976 | 60 | 0.5–5.0 | Kaplinghat, Tulin & Yu 2016 |
| NGC 1560 | 55 | 1.0–8.0 | Kaplinghat, Tulin & Yu 2016 |
| IC 2574 | 50 | 0.3–5.0 | Kaplinghat, Tulin & Yu 2016 |
| NGC 720 (group) | 250 | 0.1–1.5 | Kaplinghat, Tulin & Yu 2016 |
| NGC 1332 (group) | 280 | 0.05–1.0 | Kaplinghat, Tulin & Yu 2016 |
| Abell 611 | 1200 | 0.02–0.3 | Kaplinghat, Tulin & Yu 2016 |
| Abell 2537 | 1100 | 0.03–0.4 | Kaplinghat, Tulin & Yu 2016 |
| Diverse RC band | 40 | 0.5–10.0 | Kel'ner, Kirchner, Peters & Yu 2017 |
| Bullet Cluster | 4700 | < 1.25 | Randall et al. 2008 |
| 72 cluster mergers | 1000 | < 0.47 | Harvey et al. 2015 |
| TBTF dwarfs | 30 | 0.5–5.0 | Elbert et al. 2015 |

**Primary reference:**
> Kaplinghat, Tulin & Yu (2016), *Phys. Rev. Lett.* 116, 041302.
> DOI: [10.1103/PhysRevLett.116.041302](https://doi.org/10.1103/PhysRevLett.116.041302)

---

## 2. Fornax dSph Halo Parameters

**File:** `data/observational_database.json` → key `"fornax_halo"`

- M₂₀₀ = 3.16 × 10⁹ M☉, c₂₀₀ = 18, σ_v = 11.7 km/s, r_half = 710 pc
- Source: Read, Walker & Steger (2019), *MNRAS* 484, 1401.
- DOI: [10.1093/mnras/sty3497](https://doi.org/10.1093/mnras/sty3497)

---

## 3. Walker+2009 Fornax Velocity Dispersion Profile

**File:** `data/observational_database.json` → key `"walker2009_fornax"`

- 9 radial bins (R = 100–1800 pc), σ_los from 2633 member stars
- Source: Walker, Mateo, Olszewski et al. (2009), *ApJ* 704, 1274.
- DOI: [10.1088/0004-637X/704/2/1274](https://doi.org/10.1088/0004-637X/704/2/1274)

---

## 4. Cosmological Constants (Planck 2018)

**File:** `data/observational_database.json` → key `"cosmological_constants"`

- h = 0.674, Ω_m h² = 0.1200, N_eff = 2.99 ± 0.17
- Source: Planck Collaboration (2018), *A&A* 641, A6.
- DOI: [10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910)

---

## 5. Cluster Merger (SIDM Upper Bound)

- σ/m < 0.47 cm²/g at 95% CL from 72 cluster mergers
- Source: Harvey, Massey, Kitching et al. (2015), *Science* 347, 1462.
- DOI: [10.1126/science.1261381](https://doi.org/10.1126/science.1261381)

---

## 6. Bullet Cluster (Hard Bound)

- σ/m < 1.25 cm²/g from Bullet Cluster mass segregation
- Source: Randall, Markevitch, Clowe et al. (2008), *ApJ* 679, 1173.
- DOI: [10.1086/587859](https://doi.org/10.1086/587859)

---

## 7. Dark Energy Equation of State (DESI 2024)

- w₀ = -0.827 ± 0.063 (1σ), w_a = -0.75 ± 0.29
- Used in Chapter 3, §Observational Signatures
- Source: DESI Collaboration (2024), *arXiv:2404.03002*
- URL: [https://arxiv.org/abs/2404.03002](https://arxiv.org/abs/2404.03002)

---

## 8. Computed Scan Results (this work)

These CSV files were produced by running the Chapter 1 code and are included
for reproducibility. They are **not** independently downloaded.

| File | Contents | Produced by |
|------|----------|-------------|
| `all_viable_representative.csv` | Representative sample of SIDM-viable parameter space | `chapter1_sidm/smart_scan.py` |
| `sidm_relic_viable_points.csv` | Points satisfying both SIDM + relic density | `chapter1_sidm/boltzmann_correction.py` |
| `benchmark_points.csv` | Explored benchmark points from grid scan | `chapter1_sidm/v22_raw_scan_fast.py` |
| `chi2_fit_results.csv` | χ² fit to 13 observational systems | `chapter1_sidm/pipeline.py` |
| `mcmc_checkpoint.npy` | MCMC sampler state (emcee, 32 walkers × 5000 steps) | `chapter1_sidm/opusB_run_mcmc.py` |

---

## 9. How Observational Data is Accessed in Code

All Chapter 1 scripts import data via:

```python
from core.global_config import GC

obs = GC.observations()          # 13-system constraint table
fornax = GC.fornax_halo()        # Fornax halo parameters
walker = GC.walker2009_fornax()  # Fornax σ_los profile
cosmo  = GC.cosmological_constants()  # Planck 2018
```

The `core/global_config.py` loader reads from `data/observational_database.json`.
If running scripts from `chapter1_sidm/` directly (outside the full repo),
point the config path to `../data/observational_database.json`.

---

## 10. No External Downloads Required

All numbers in this repository come from published papers (tabulated above).
No web scraping, no private datasets, no proprietary data.
The full paper can be reproduced from a clean install with only:

```bash
pip install -r requirements.txt
python chapter1_sidm/opusB_run_mcmc.py   # ~9h on CPU, or ~1h on GPU with numba
```
