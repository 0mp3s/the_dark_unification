"""
config.py — Centralized constants and benchmarks for Chapter 3
==============================================================

All physical constants, benchmark parameters, and observational bounds
used across Chapter 3 scripts. Single source of truth.

Usage:
    from config import GeV, MeV, eV, M_Pl, H_0, rho_L, MAP, BENCHMARKS
"""

import numpy as np

# =============================================================================
# Unit definitions (natural units, GeV = 1)
# =============================================================================
GeV = 1.0
MeV = 1e-3 * GeV
eV  = 1e-9 * GeV
keV = 1e-6 * GeV

# =============================================================================
# Fundamental physical constants
# =============================================================================
M_Pl     = 2.435e18 * GeV       # reduced Planck mass
M_Pl_full = 1.2209e19 * GeV     # non-reduced Planck mass (= M_Pl * sqrt(8π))
H_0      = 1.44e-42 * GeV       # Hubble constant (67.4 km/s/Mpc)
rho_L    = 2.58e-47 * GeV**4    # observed dark energy density ρ_Λ
T_BBN    = 1.0 * MeV            # BBN temperature
T_CMB_eV = 2.7255 * 8.617e-5    # CMB temperature in eV
T_CMB    = T_CMB_eV * eV        # CMB temperature in GeV

# Hubble decomposition (used by background_cosmology.py)
H_100_GEV   = 2.1332e-42        # 100 km/s/Mpc in GeV
H0_PLANCK_KMS = 67.4            # km/s/Mpc
OMEGA_B_H2  = 0.02237           # baryon density (Planck 2018)
N_EFF       = 3.044             # effective number of neutrino species

# Electroweak
m_H       = 125.1 * GeV         # Higgs boson mass
v_EW      = 246.22 * GeV        # electroweak VEV
Gamma_H_SM = 4.07e-3 * GeV      # SM Higgs total width

# Speed of light
c_kms     = 299792.458          # km/s

# Relic density conversion
OMEGA_FACTOR = 2.742e8          # s₀/(ρ_c/h²) in GeV⁻¹

# =============================================================================
# Dark sector structural parameters
# =============================================================================
g_star_BBN = 10.75              # SM g* at BBN
g_gluons   = 6                  # SU(2)_d: (N²-1)=3 gluons × 2 pol.
g_phi_boson = 1                 # real scalar mediator
Lambda_d   = 2.0e-12 * GeV     # dark QCD scale (≈ 2 meV)

# =============================================================================
# A₄ relic angle
# =============================================================================
theta_relic = np.arctan(1.0 / 3.0)    # 18.43°, from g_p/g_s = 1/3
sin2_theta  = np.sin(theta_relic)**2   # = 1/10
cos2_theta  = np.cos(theta_relic)**2   # = 9/10
AB_RATIO    = 39.0 / 5.0              # A/B from A₄ CG coefficients

# =============================================================================
# Benchmark points (from MCMC posterior, Paper 1)
# =============================================================================
# All masses in GeV, dimensionless coupling α_d
BENCHMARKS = {
    "BP1":       {"m_chi": 20.69 * GeV,  "m_phi": 11.34 * MeV, "alpha": 1.048e-3},
    "BP9":       {"m_chi": 42.53 * GeV,  "m_phi": 10.92 * MeV, "alpha": 2.165e-3},
    "BP16":      {"m_chi": 63.81 * GeV,  "m_phi": 11.78 * MeV, "alpha": 3.253e-3},
    "MAP":       {"m_chi": 94.07 * GeV,  "m_phi": 11.10 * MeV, "alpha": 5.734e-3},
    "MAP_relic": {"m_chi": 85.84 * GeV,  "m_phi": 15.35 * MeV, "alpha": 5.523e-3},
}

# Convenience: MAP benchmark as top-level variables
MAP = BENCHMARKS["MAP"]

# Derived Yukawa coupling from MAP
y_MAP = np.sqrt(4 * np.pi * MAP["alpha"] / cos2_theta)

# =============================================================================
# Experimental bounds
# =============================================================================
BR_inv_limit = 0.11             # H → invisible BR limit
sigma_Planck = 0.17             # Planck+BAO 1σ on N_eff (2.99 ± 0.17)
sigma_Simons = 0.05             # Simons Observatory σ(N_eff)
sigma_CMBS4  = 0.027            # CMB-S4 σ(N_eff)
beta_max     = 0.066            # fifth force constraint (Amendola+ 2020)

# QCD reference scales
Lambda_QCD         = 200.0 * MeV    # Λ_QCD (MS-bar)
Lambda_QCD_lattice = 210.0 * MeV    # lattice QCD
T_QCD_crossover    = 155.0 * MeV    # Borsanyi et al. 2010

# Known SM particle masses (MeV, for qcd_scale_coincidence comparisons)
m_pion_neutral = 134.97 * MeV
m_pion_charged = 139.57 * MeV
m_muon         = 105.66 * MeV

# =============================================================================
# SM entropy d.o.f. g*_S(T) interpolation table
# Based on Borsanyi et al. (2016) lattice QCD + standard thresholds
# =============================================================================
_SM_T = np.array([
    1e-5,  5e-4,  1e-3,  5e-3,  0.01,  0.02,
    0.10,  0.15,  0.17,  0.20,  0.30,  0.50,
    1.0,   2.0,   4.0,   5.0,  80.0, 170.0,
    300.0, 1e4,   1e8,   1e12,  1e16
])
_SM_g = np.array([
    3.91,  3.91, 10.75, 10.75, 10.75, 10.75,
    17.25, 25.0,  40.0, 61.75, 61.75, 61.75,
    61.75, 75.75, 86.25, 86.25, 96.25, 106.75,
    106.75, 106.75, 106.75, 106.75, 106.75
])

def g_star_S(T_GeV):
    """SM entropy d.o.f. g*_S(T) via log-linear interpolation."""
    return np.interp(np.log(T_GeV), np.log(_SM_T), _SM_g)
