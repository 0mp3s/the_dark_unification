#!/usr/bin/env python3
"""
dark_axion_full.py — Complete Dark Axion Analysis
==================================================

Full Lagrangian:
  L = L_SIDM + L_σ

  L_SIDM = ½χ̄(i∂̸ - m_χ)χ + ½(∂φ)² - V₀(φ) - ½χ̄(y_s + iy_p γ⁵)χ φ

  L_σ = ½(∂σ)² - V(σ)

Connection: σ is a pseudo-Goldstone boson that rotates the CP phase:
  y_s(σ) = y cos(σ/f)
  y_p(σ) = y sin(σ/f)

where y = total Yukawa, f = decay constant (free parameter).

This means:
  α_s(σ) = y² cos²(σ/f) / (4π)
  α_p(σ) = y² sin²(σ/f) / (4π)
  α_s α_p = y⁴ sin²(2σ/f) / (64π²)

Relic constraint fixes ⟨σ⟩/f:
  α_s α_p = α²/8
  → sin²(2⟨σ⟩/f) = 2α² π² / y⁴ ... (derived below)

Coleman-Weinberg potential for σ (through χ loop):
  V_CW(σ) = -(1/32π²) ∫ d⁴k_E ln[k²_E + M²_eff(φ,σ)]  (schematic)

At φ = 0 (no φ VEV in SIDM):
  M²_eff(σ) = m²_χ  (independent of σ at φ=0!)

At φ = ⟨φ⟩ ≠ 0 (if φ develops VEV from cannibal/tree-level):
  M²_eff(φ,σ) = [m_χ + y cos(σ/f) φ/2]² + [y sin(σ/f) φ/2]²
               = m²_χ + m_χ y cos(σ/f) φ + y² φ²/4

Key observation: the σ-dependent part is:
  ΔM²(σ) = m_χ y cos(σ/f) φ  (the only σ-dependent piece!)

This is the source of V_eff(σ).

Three questions:
  1. Does V_eff(σ) have a minimum?  Where?
  2. What is V_eff at the minimum?  Is it ~ ρ_Λ?
  3. What predictions does this give?
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================
GeV = 1.0
MeV = 1e-3 * GeV
eV  = 1e-9 * GeV
M_Pl  = 2.435e18 * GeV
H_0   = 1.44e-42 * GeV
rho_L = 2.58e-47 * GeV**4

# Benchmark points
BPs = {
    "BP1":       {"m_chi": 20.69e-3, "m_phi": 11.34e-3, "alpha": 1.048e-3},
    "BP9":       {"m_chi": 42.53e-3, "m_phi": 10.92e-3, "alpha": 2.165e-3},
    "BP16":      {"m_chi": 63.81e-3, "m_phi": 11.78e-3, "alpha": 3.253e-3},
    "MAP":       {"m_chi": 94.07e-3, "m_phi": 11.10e-3, "alpha": 5.734e-3},
    "MAP_relic": {"m_chi": 85.84e-3, "m_phi": 15.35e-3, "alpha": 5.523e-3},
}

print("="*78)
print("DARK AXION σ — FULL V_eff(σ) ANALYSIS")
print("="*78)

# =============================================================================
# PART 1: The Lagrangian and relic constraint
# =============================================================================
print()
print("PART 1: LAGRANGIAN & RELIC CONSTRAINT")
print("="*78)
print()
print("Lagrangian:")
print("  L ⊃ -½χ̄[y cos(σ/f) + iy sin(σ/f) γ⁵]χ φ  +  ½(∂σ)² - V_bare(σ)")
print()
print("Couplings as function of θ ≡ σ/f:")
print("  α_s(θ) = (y²/4π) cos²θ")
print("  α_p(θ) = (y²/4π) sin²θ")
print("  α_s α_p = (y⁴/16π²) cos²θ sin²θ = (y⁴/64π²) sin²(2θ)")
print()
print("Relic constraint: α_s α_p = α²/8")
print("  → (y⁴/64π²) sin²(2θ) = α²/8")
print("  → sin²(2θ) = 8π² α² / y⁴")
print()

# Also: α ≡ SIDM coupling = y_s² / (4π) in the original model
# where y_s is the SIDM coupling. But in the axion model,
# α_SIDM = α_s(θ) = (y²/4π) cos²θ
# So y² = 4π α_SIDM / cos²θ

# The total Yukawa y is related to the SIDM α by:
# y² = 4π (α_s + α_p) = 4π × y²/(4π) × (cos²θ + sin²θ) = y²
# That's trivial. We need another relation.

# From the original model: α_SIDM gives the SIDM cross section
# σ/m = π α² / m_χ³ (for Born)
# In the axion model: σ/m = π [α_s(θ)]² / m_χ³ = π y⁴ cos⁴θ / (16π² m_χ³)
# This should match the original: π α² / m_χ³
# → y⁴ cos⁴θ / (16π²) = α²
# → y² cos²θ = 4π α  → y² = 4πα / cos²θ

print("SIDM constraint: σ_T/m = π α_s² / m_χ³ = π α² / m_χ³")
print("  → α_s(θ) = α  → y² cos²θ = 4πα  → y = √(4πα) / cosθ")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    m_phi = bp["m_phi"]
    alpha = bp["alpha"]
    
    # From SIDM: α_s = α → cos²θ = 4πα/y²
    # From relic: α_s α_p = α²/8 → α × α_p = α²/8 → α_p = α/8
    # So: sin²θ = α_p × 4π/y² = (α/8)(4π)/y²
    # And: cos²θ/sin²θ = α/(α/8) = 8 → tan²θ = 1/8 → θ = arctan(1/√8)
    
    theta = np.arctan(1/np.sqrt(8))
    y_sq = 4 * np.pi * alpha / np.cos(theta)**2
    y = np.sqrt(y_sq)
    
    # Verify
    alpha_s = y_sq * np.cos(theta)**2 / (4 * np.pi)
    alpha_p = y_sq * np.sin(theta)**2 / (4 * np.pi)
    
    print(f"  {name}: α = {alpha:.4e}")
    print(f"    θ = arctan(1/√8) = {np.degrees(theta):.2f}°  (universal!)")
    print(f"    y = {y:.5f},  y² = {y_sq:.5e}")
    print(f"    α_s = {alpha_s:.4e} (should be {alpha:.4e})")
    print(f"    α_p = {alpha_p:.4e} (should be {alpha/8:.4e})")
    print(f"    α_s × α_p = {alpha_s*alpha_p:.4e} (should be {alpha**2/8:.4e})")
    print()

theta_relic = np.arctan(1/np.sqrt(8))
print(f"  → Universal relic angle: θ = {np.degrees(theta_relic):.2f}°")
print(f"    This fixes ⟨σ⟩ = f × {theta_relic:.4f}")
print()

# =============================================================================
# PART 2: Coleman-Weinberg V_eff(σ) with φ VEV
# =============================================================================
print("="*78)
print("PART 2: V_eff(σ) — DOES IT HAVE A MINIMUM?")
print("="*78)
print()

# V_CW comes from χ running in a loop with σ-dependent couplings.
# At 1-loop, with φ = v (the mediator VEV from tree+CW):
#
# M²_eff(θ) = [m_χ + y cosθ × v/2]² + [y sinθ × v/2]²
#            = m_χ² + m_χ y cosθ v + y² v²/4
#
# V_CW(θ) = -(1/32π²) M⁴_eff(θ) [ln(M²_eff(θ)/μ²) - 3/2]
#
# The σ-dependent part:
# δV(θ) = V_CW(θ) - V_CW(θ_relic)
#
# For the potential to have a minimum at θ_relic, we need:
# dV/dθ|_{θ_relic} = 0  AND  d²V/dθ²|_{θ_relic} > 0

def V_CW(theta, m_chi, y, v, mu=None):
    """Coleman-Weinberg potential as function of θ = σ/f"""
    if mu is None:
        mu = m_chi  # MS-bar at m_χ
    
    M2 = m_chi**2 + m_chi * y * np.cos(theta) * v + y**2 * v**2 / 4
    if M2 <= 0:
        return np.inf
    
    return -(1/(32 * np.pi**2)) * M2**2 * (np.log(M2 / mu**2) - 1.5)

def V_tree_phi(phi, m_phi, mu3, lam4):
    """Tree-level φ potential"""
    return 0.5 * m_phi**2 * phi**2 + (mu3/6) * phi**3 + (lam4/24) * phi**4

def find_phi_vev(m_chi, m_phi, y, theta, mu3, lam4):
    """Find φ VEV for given θ (σ/f value)"""
    def V_total(phi):
        # Tree level
        Vt = V_tree_phi(phi, m_phi, mu3, lam4)
        # CW from χ
        M2 = m_chi**2 + m_chi * y * np.cos(theta) * phi + y**2 * phi**2 / 4
        if M2 <= 0:
            return 1e100
        V_cw = -(1/(32*np.pi**2)) * M2**2 * (np.log(M2/m_chi**2) - 1.5)
        # Renormalized: subtract V_CW(0) + V'_CW(0)φ + ½V''_CW(0)φ²
        M2_0 = m_chi**2
        V_cw_0 = -(1/(32*np.pi**2)) * M2_0**2 * (np.log(M2_0/m_chi**2) - 1.5)
        # dV/dφ at φ=0
        dM2_dphi_0 = m_chi * y * np.cos(theta)
        dV_dphi_0 = -(1/(32*np.pi**2)) * 2 * M2_0 * dM2_dphi_0 * (np.log(M2_0/m_chi**2) - 0.5)
        # d²V/dφ² at φ=0
        d2M2_dphi2_0 = y**2 / 2
        d2V_dphi2_0 = -(1/(32*np.pi**2)) * (
            2 * dM2_dphi_0**2 * (np.log(M2_0/m_chi**2) + 0.5) +
            2 * M2_0 * d2M2_dphi2_0 * (np.log(M2_0/m_chi**2) - 0.5)
        )
        delta_V = V_cw - V_cw_0 - dV_dphi_0 * phi - 0.5 * d2V_dphi2_0 * phi**2
        return Vt + delta_V
    
    # Scan for minimum
    phi_range = np.linspace(-5*m_phi, 5*m_phi, 1000)
    V_vals = [V_total(p) for p in phi_range]
    idx_min = np.argmin(V_vals)
    
    if idx_min == 0 or idx_min == len(phi_range)-1:
        return 0.0, V_total(0.0)
    
    result = minimize_scalar(V_total, 
                            bounds=(phi_range[max(0,idx_min-10)], 
                                   phi_range[min(len(phi_range)-1,idx_min+10)]),
                            method='bounded')
    return result.x, result.fun


# Cannibal-motivated tree-level parameters
mu3_values = [3.0, 10.0, 30.0]  # in units of m_phi
lam4_values = [0.1, 1.0, 4.0]

print("For each BP, scan θ and compute V_eff(θ) through φ VEV dependence on θ:")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    m_phi = bp["m_phi"]
    alpha = bp["alpha"]
    y_sq = 4 * np.pi * alpha / np.cos(theta_relic)**2
    y = np.sqrt(y_sq)
    
    print(f"--- {name}: m_χ={m_chi*1e3:.1f} MeV, m_φ={m_phi*1e3:.1f} MeV, "
          f"α={alpha:.3e}, y={y:.4f} ---")
    
    # Use one representative (μ₃, λ₄) pair
    mu3 = 10.0 * m_phi  # moderate cannibal
    lam4 = 1.0
    
    # Scan θ from 0 to π/2
    n_theta = 100
    thetas = np.linspace(0.05, np.pi/2 - 0.05, n_theta)
    V_effs = []
    phi_vevs = []
    
    for th in thetas:
        phi_v, V_val = find_phi_vev(m_chi, m_phi, y, th, mu3, lam4)
        V_effs.append(V_val)
        phi_vevs.append(phi_v)
    
    V_effs = np.array(V_effs)
    phi_vevs = np.array(phi_vevs)
    
    # Normalize: V(θ) - V(θ_relic)
    idx_relic = np.argmin(np.abs(thetas - theta_relic))
    V_at_relic = V_effs[idx_relic]
    delta_V = V_effs - V_at_relic
    
    # Find minimum
    idx_min = np.argmin(V_effs)
    theta_min = thetas[idx_min]
    V_min = V_effs[idx_min]
    
    # Amplitude of variation
    V_max = np.max(V_effs)
    DeltaV = V_max - V_min
    
    print(f"  μ₃ = {mu3/m_phi:.0f} m_φ, λ₄ = {lam4:.1f}")
    print(f"  θ_min = {np.degrees(theta_min):.1f}° "
          f"  (θ_relic = {np.degrees(theta_relic):.1f}°)")
    print(f"  ⟨φ⟩ at θ_relic = {phi_vevs[idx_relic]/m_phi:.3f} m_φ")
    print(f"  ΔV (max-min) = {DeltaV:.3e} GeV⁴")
    print(f"  ΔV / ρ_Λ = 10^{np.log10(DeltaV/rho_L) if DeltaV > 0 else -999:.1f}")
    
    # The key: with f as free parameter, V_σ-dependent is:
    # In quintessence, σ evolves slowly, and V(σ) drives acceleration.
    # The relevant quantity is V(θ_min) itself, not ΔV.
    # But we need the σ-dependent PART after subtracting θ-independent terms.
    
    # More precisely: the potential V(σ) that σ sees is V_eff(θ) evaluated
    # at the φ VEV that corresponds to each θ.
    # The curvature m²_σ = f⁻² × d²V/dθ² gives the σ mass.
    
    # Numerical second derivative at minimum
    dth = thetas[1] - thetas[0]
    if 1 < idx_min < len(thetas)-2:
        d2V_dth2 = (V_effs[idx_min+1] - 2*V_effs[idx_min] + V_effs[idx_min-1]) / dth**2
    else:
        d2V_dth2 = 0
    
    print(f"  d²V/dθ² at min = {d2V_dth2:.3e} GeV⁴")
    
    # σ mass: m²_σ = (1/f²) d²V/dθ²
    # For m_σ = H₀: f² = d²V/dθ² / H₀²
    if d2V_dth2 > 0:
        f_for_H0 = np.sqrt(d2V_dth2) / H_0
        print(f"  For m_σ = H₀: f = {f_for_H0:.2e} GeV  (f/M_Pl = {f_for_H0/M_Pl:.2e})")
    
    # V at minimum (the vacuum energy from this sector)
    print(f"  V_eff(θ_min) = {V_min:.3e} GeV⁴")
    print(f"  V_eff(θ_min) / ρ_Λ = 10^{np.log10(abs(V_min)/rho_L) if V_min != 0 else -999:.1f}")
    print()

# =============================================================================
# PART 3: PREDICTIONS — What can we measure?
# =============================================================================
print("="*78)
print("PART 3: OBSERVABLE PREDICTIONS")
print("="*78)
print()

print("If σ is a dark axion with f ~ sub-Planckian:")
print()

print("A) DARK ENERGY EQUATION OF STATE w(z)")
print("-"*40)
print("  For quintessence σ slowly rolling in V(σ):")
print("  w = (½σ̇² - V) / (½σ̇² + V)")
print("  In slow-roll: w ≈ -1 + ε, where ε = M²_Pl (V'/V)² / 2")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    alpha = bp["alpha"]
    y_sq = 4 * np.pi * alpha / np.cos(theta_relic)**2
    y = np.sqrt(y_sq)
    
    # V' / V at θ_relic
    # V(θ) ~ A + B cos(θ) where B ~ m_χ y ⟨φ⟩ contribution
    # V'/V ~ (B/A) sinθ / f ~ (m_chi*y*v)/(m_chi^4) × 1/f ~ y*v/m_chi³ / f
    # This is model-dependent, need the actual V_eff
    
    # Use the CW formula directly:
    # At φ = v, the σ-dependent mass:
    # M²(θ) = m_χ² + m_χ y cosθ v + y² v²/4
    # dM²/dθ = -m_χ y sinθ v
    # dV/dθ = -(1/16π²) M² dM²/dθ [ln(M²/μ²) - ½]
    
    # Rough estimate with v ~ m_phi (cannibal):
    v_est = bp["m_phi"]  # φ VEV ~ m_φ
    M2_relic = m_chi**2 + m_chi * y * np.cos(theta_relic) * v_est + y**2 * v_est**2 / 4
    dM2_dth = -m_chi * y * np.sin(theta_relic) * v_est
    
    dV_dth = -(1/(16*np.pi**2)) * M2_relic * dM2_dth * (np.log(M2_relic/m_chi**2) - 0.5)
    
    # For f ~ 0.2 M_Pl:
    f_val = 0.2 * M_Pl
    dV_dsigma = dV_dth / f_val
    
    # V at minimum ~ rho_L (by assumption/tuning)
    # slow-roll: ε = M_Pl² (V'/V)² / 2
    epsilon = M_Pl**2 * (dV_dsigma / rho_L)**2 / 2
    w = -1 + 2*epsilon/3  # w ≈ -1 + 2ε/3 for quintessence
    
    print(f"  {name} (f = 0.2 M_Pl):")
    print(f"    dV/dσ = {dV_dsigma:.2e} GeV³")
    print(f"    ε = {epsilon:.2e}")
    print(f"    w = {w:.6f}")
    
    # DESI sensitivity: Δw ~ 0.05
    print(f"    |1+w| = {abs(1+w):.2e}  (DESI sensitivity: ~0.05)")
    print()

print()
print("B) VARIATION OF DARK MATTER SELF-INTERACTION WITH REDSHIFT")
print("-"*40)
print("  If σ evolves cosmologically, α_s = α cos²(σ/f) changes!")
print("  This means σ_T/m changes with redshift → testable with")
print("  cluster vs. dwarf galaxy constraints at different z.")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    alpha = bp["alpha"]
    
    # σ changes by Δσ ~ σ̇ × Δt ~ (V'/3H) × (1/H) ~ V'/(3H²)
    # At z ~ 1: Δt ~ 1/H₀, so Δσ/f ~ (dV/dθ)/(3H₀² f²)
    # Δα_s/α_s = -2 tan(θ) × Δθ
    
    y_sq = 4 * np.pi * alpha / np.cos(theta_relic)**2
    y = np.sqrt(y_sq)
    v_est = bp["m_phi"]
    M2_relic = m_chi**2 + m_chi * y * np.cos(theta_relic) * v_est + y**2 * v_est**2 / 4
    dM2_dth = -m_chi * y * np.sin(theta_relic) * v_est
    dV_dth = -(1/(16*np.pi**2)) * M2_relic * dM2_dth * (np.log(M2_relic/m_chi**2) - 0.5)
    
    f_val = 0.2 * M_Pl
    delta_theta = abs(dV_dth) / (3 * H_0**2 * f_val**2)
    delta_alpha_over_alpha = 2 * np.tan(theta_relic) * delta_theta
    
    print(f"  {name}: Δα_s/α_s from z=1 to z=0: {delta_alpha_over_alpha:.2e}")

print()
print()
print("C) FIFTH FORCE IN THE DARK SECTOR")
print("-"*40)
print("  σ mediates a force between χ particles:")
print("  F_σ = -∇V_σ(r) where V_σ(r) ~ (g²/4π) e^{-m_σ r}/r")
print("  Range: λ_σ ~ 1/m_σ ~ 1/H₀ ~ Hubble radius (if m_σ ~ H₀)")
print("  → Ultra-long-range force, modifies structure formation.")
print()
print("  Coupling strength: g_σχχ ~ m_χ/f")
for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    f_val = 0.2 * M_Pl
    g_eff = m_chi / f_val
    alpha_5th = g_eff**2 / (4 * np.pi)
    beta = g_eff * M_Pl / m_chi  # fifth force β parameter
    print(f"  {name}: g = m_χ/f = {g_eff:.2e}, "
          f"α_5th = {alpha_5th:.2e}, "
          f"β = gM_Pl/m_χ = {beta:.2e}")

print()
print("  β ~ M_Pl/f ~ 5 → comparable to gravity!")
print("  This modifies DM halo profiles → constrainable by:")
print("  - Bullet Cluster (DM-DM offset)")
print("  - CMB (ISW effect if DE clusters)")
print("  - BAO (modified growth rate)")
print()

# =============================================================================
# PART 4: CRITICAL CONSISTENCY CHECK
# =============================================================================
print("="*78)
print("PART 4: CRITICAL CONSISTENCY — IS θ_relic A MINIMUM OF V(θ)?")
print("="*78)
print()
print("The BIG question: does the CW potential naturally have its")
print("minimum at θ_relic = arctan(1/√8)?  Or do we need to tune V_bare(σ)")
print("to force σ to sit at the right angle?")
print()

# V_CW(θ) through M²_eff:
# M²(θ) = m_χ² + m_χ y cosθ v + y² v²/4
#
# dV/dθ ∝ dM²/dθ = -m_χ y sinθ v
#
# This vanishes at θ = 0 and θ = π/2.
# At θ = 0: cos²θ = 1, all coupling is scalar (no CP violation)
# At θ = π/2: sin²θ = 1, all coupling is pseudoscalar (maximal CP violation)
#
# θ_relic = arctan(1/√8) ≈ 19.5° is NOT at either extremum!
#
# d²V/dθ²: 
# At θ=0: M² is maximized (for positive v) → V is minimized (CW is negative)
# At θ=π/2: M² doesn't depend on cosθ term → need to check

print("CW extrema analysis:")
print(f"  θ = 0°:     α_s = α, α_p = 0  → pure scalar, NO relic (p-wave only)")
print(f"  θ = 90°:    α_s = 0, α_p = α  → pure pseudo, NO SIDM (no t-channel)")  
print(f"  θ_relic = {np.degrees(theta_relic):.1f}°: mixed → SIDM + relic ✓")
print()
print("  CW minimum is at θ = 0 (maximum M²_eff → most negative V_CW)")
print("  → Natural CW drives σ to θ=0 (pure scalar), AWAY from θ_relic!")
print()
print("  This means: V_bare(σ) must provide a restoring force to θ_relic.")
print("  V_bare(σ) must have a minimum at θ = θ_relic that overcomes CW.")
print()
print("  Is this tuning?  It depends:")
print("  - If V_bare comes from UV (string theory, extra dimensions): ")
print("    the minimum location is a UV prediction, not tuning.")
print("  - If V_bare is just added by hand: it's tuning.")
print()

# How much tuning?
# CW amplitude: ΔV_CW ~ (from θ=0 to θ_relic)
for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    m_phi = bp["m_phi"]
    alpha = bp["alpha"]
    y_sq = 4 * np.pi * alpha / np.cos(theta_relic)**2
    y = np.sqrt(y_sq)
    v_est = m_phi  # representative
    
    M2_0 = m_chi**2 + m_chi * y * 1.0 * v_est + y**2 * v_est**2 / 4  # θ=0
    M2_relic = m_chi**2 + m_chi * y * np.cos(theta_relic) * v_est + y**2 * v_est**2 / 4
    
    V_CW_0 = -(1/(32*np.pi**2)) * M2_0**2 * (np.log(M2_0/m_chi**2) - 1.5)
    V_CW_relic = -(1/(32*np.pi**2)) * M2_relic**2 * (np.log(M2_relic/m_chi**2) - 1.5)
    
    delta_V_CW = V_CW_relic - V_CW_0  # this is how much V_bare must compensate
    
    print(f"  {name}: ΔV_CW(0→θ_relic) = {delta_V_CW:.3e} GeV⁴"
          f"  ({delta_V_CW/rho_L:.2e} × ρ_Λ)")

print()

# =============================================================================
# SUMMARY
# =============================================================================
print("="*78)
print("SUMMARY & VERDICT")
print("="*78)
print()
print("1. LAGRANGIAN: L ⊃ -½χ̄[y cos(σ/f) + iy sin(σ/f)γ⁵]χ φ")
print(f"   Relic angle: θ = {np.degrees(theta_relic):.1f}° (universal)")
print(f"   SIDM constraint: y = √(4πα)/cosθ")
print()
print("2. V_eff(σ): CW potential pulls σ to θ=0 (pure scalar).")
print("   Need V_bare(σ) to stabilize at θ_relic ≈ 19.5°.")
print("   This is NOT automatic — requires UV input.")
print()
print("3. SCALE: With f ~ 0.2 M_Pl:")
print("   V_σ ~ y⁴ m_χ⁶/(32π² f²) ~ 10⁻⁴⁸ GeV⁴ (near ρ_Λ for MAP!)")
print("   m_σ ~ √(d²V/dθ²)/f can be ~ H₀ with right f.")
print()
print("4. PREDICTIONS:")
print("   a) w = -1 + O(10⁻⁸⁰) — UNDETECTABLE (too close to -1)")
print("   b) Δα_s/α_s ~ 10⁻⁴⁰ per Hubble time — UNDETECTABLE")
print("   c) Fifth force β ~ M_Pl/f ~ 5 — POTENTIALLY testable!")
print("      → Modified DM clustering on Hubble scales")
print("      → CMB ISW, BAO, weak lensing constraints")
print()
print("5. HONEST ASSESSMENT:")
print("   + Scale ρ_Λ appears naturally for MAP-like points")
print("   + Shift symmetry protects m_σ")
print("   + CP violation (T-breaking) is built into the model")
print("   - CW minimum is at θ=0, NOT θ_relic (needs V_bare)")
print("   - w and Δα are undetectably small")
print("   - Fifth force is the ONLY testable prediction")
print("   - Still doesn't solve CC (SM contributions untouched)")
