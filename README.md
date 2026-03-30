# The Dark Unification

**A unified framework connecting self-interacting dark matter, its quantum path-integral foundations, and dark energy as the T-breaking component of the same interaction.**

> *"Dark energy is not a separate phenomenon — it is the T-violating projection of dark matter,  
> exactly as magnetism is the T-violating projection of electrostatics."*

---

## Author

P., Omer — Independent Researcher

---

## Structure

This repository contains the unified theoretical framework spanning three interconnected chapters:

### Chapter 1 — Secluded Majorana SIDM
**Self-Interacting Dark Matter via a Secluded Scalar Mediator**

A Majorana fermion DM candidate $\chi$ interacts via a real scalar mediator $\phi$ through a Yukawa coupling. The Variable Phase Method (VPM) computes velocity-dependent transfer cross sections exactly. A complete numerical pipeline covers: SIDM cross sections, relic density (Boltzmann), and confrontation with astrophysical observations spanning dwarfs to clusters.

- 122 viable benchmark points in $(m_\chi, m_\phi, \alpha)$ space  
- MCMC posterior: MAP at $(94\,\text{GeV},\ 11\,\text{MeV},\ 5.7\times10^{-3})$  
- Fornax GC survival: 14/15 at MAP; Read+2019 bound: 48% of island survives  
- Source: [`Secluded-Majorana-SIDM`](https://github.com/0mp3s/Secluded-Majorana-SIDM) | DOI: [10.5281/zenodo.19225823](https://doi.org/10.5281/zenodo.19225823)

### Chapter 2 — Path Integral Foundations
**From $\mathcal{L}$ to $H_0$: A 7-Layer Derivation**

A complete first-principles derivation of all Chapter 1 results from the generating functional $Z[J]$, plus an extension to the Coleman-Weinberg potential, finite-T Matsubara formalism, and dark QCD misalignment:

$$\mathcal{L}_{SIDM} \xrightarrow{\int\mathcal{D}} Z[J] \xrightarrow{\text{Gaussian}} \Delta_F,\, S_F \xrightarrow{\text{tree}} V(r) \xrightarrow{\text{det}} \sigma_T(v) \xrightarrow{\langle\sigma v\rangle} \Omega h^2$$

$$\mathcal{L}_{SIDM} \xrightarrow{1\text{-loop}} V_{CW}(\theta) \xrightarrow{A_4} \theta_{relic} \xrightarrow{\text{dark QCD}} \Omega_\sigma = 0.69 \xrightarrow{\text{Friedmann}} H_0 = 67.4\,\text{km/s/Mpc}$$

Key result: **VPM is exactly the fluctuation determinant of the path integral** (PI-6, proven). No approximation. The Hubble constant emerges from $V_{eff}(\sigma_0) = \rho_\Lambda$ via dark QCD confinement at $\Lambda_d \sim 2.05\,\text{meV}$.

- Source: `The_Lagernizant_integral_SIDM/` (within Secluded-Majorana-SIDM)

### Chapter 3 — Dark Electromagnetic Analogy
**Dark Energy as the T-Breaking Component of Dark Matter**

The CP phase $\theta = \sigma/f$ is promoted to a dynamical field. The analogy with electromagnetism is exact:

| Electromagnetism | Dark Sector |
|---|---|
| $\vec{E}$ (T-even) | $y_s = y\cos(\sigma/f)$ — SIDM clustering |
| $\vec{B}$ (T-odd) | $y_p = y\sin(\sigma/f)$ — DE acceleration |
| $F_{\mu\nu}$ | $\mathcal{Y} = ye^{i\gamma^5\sigma/f}$ |
| Lorentz boost mixes $E \leftrightarrow B$ | $\sigma$ field rotates $y_s \leftrightarrow y_p$ |

The universal relic angle $\theta_{relic} = \arctan(1/3) = 18.43°$ is fixed by $A_4$ group theory — **no free parameter**.

- Source: `dark-energy-T-breaking/` (within Secluded-Majorana-SIDM)

---

## The Unified Lagrangian

$$\boxed{\mathcal{L} = \underbrace{\frac{1}{2}\bar{\chi}(i\!\!\not\!\partial - m_\chi)\chi + \frac{1}{2}(\partial\phi)^2 - V_0(\phi)}_{\text{SIDM (Ch.1)}} - \underbrace{\frac{1}{2}\bar{\chi}\left[y\cos\!\frac{\sigma}{f} + iy\sin\!\frac{\sigma}{f}\gamma^5\right]\chi\,\phi}_{\text{unified coupling}} + \underbrace{\frac{1}{2}(\partial\sigma)^2 - V_{\text{bare}}(\sigma)}_{\text{dark axion (Ch.3)}}}$$

Chapter 2 shows that the scattering cross section and relic density both emerge as path integrals over this single Lagrangian.

---

## Repository Contents

```
the_dark_unification/
├── paper/
│   ├── main.tex          ← Unified LaTeX manuscript (3 chapters)
│   ├── main.pdf          ← (compiled output)
│   ├── references.bib    ← Bibliography
│   └── figures/          ← Figures (linked from chapter repos)
├── chapter1_sidm/        ← Symlinks / key outputs from Ch.1
├── chapter2_path_integral/  ← Symlinks / key outputs from Ch.2
├── chapter3_t_breaking/  ← Symlinks / key outputs from Ch.3
├── CITATION.cff
└── README.md
```

---

## Citation

```bibtex
@software{p_omer_2026_dark_unification,
  author    = {P., Omer},
  title     = {The Dark Unification},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Independent Researcher}
}
```

---

## Related Works

- Chapter 1 code: [Secluded-Majorana-SIDM v0.1.0](https://doi.org/10.5281/zenodo.19225823)
- License: MIT
