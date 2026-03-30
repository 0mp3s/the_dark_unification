"""
σ Mass Protection — Can m_σ be ultralight?
==========================================
The question: CW generates m_σ ~ 10⁻¹² eV. We need ~ H₀ ~ 10⁻³³ eV.
Can any known mechanism protect m_σ?

We compute m_σ from CW precisely, then test 5 protection mechanisms.
"""
import numpy as np

# ============= Constants ==============
GeV = 1.0; MeV = 1e-3; eV = 1e-9
M_Pl = 2.435e18
H_0 = 1.44e-42   # 67.4 km/s/Mpc

# MAP benchmark
m_chi = 94.07e-3   # GeV
m_phi = 11.10e-3   # GeV (mediator mass, NOT the VEV)
alpha = 5.734e-3
theta = np.arctan(1/np.sqrt(8))  # 19.47°
y = np.sqrt(4*np.pi*alpha / np.cos(theta)**2)

print("="*70)
print("  σ MASS PROTECTION MECHANISMS")
print("="*70)
print(f"\nMAP: m_χ={m_chi*1e3:.1f} MeV, m_φ={m_phi*1e3:.2f} MeV, α={alpha:.4f}")
print(f"     y={y:.4f}, θ={np.degrees(theta):.2f}°")

# ============= STEP 1: Precise CW mass ==============
print("\n" + "="*70)
print("  STEP 1: PRECISE V_CW(σ) AND m_σ")
print("="*70)

# V_CW from fermion loop (Majorana χ with mass that depends on σ via φ VEV):
# The effective mass-squared:
#   M²(σ,φ) = m_χ² + m_χ y cos(σ/f) φ + y²φ²/4
#
# CRITICAL: What is ⟨φ⟩?
# In our SIDM model, φ has NO VEV in vacuum (it's a Z₂-odd mediator).
# The χ-φ interaction is a Yukawa, not a Higgs coupling.
# φ acquires an EFFECTIVE VEV from the thermal bath: ⟨φ²⟩_T = T²/12
# But at T=0: ⟨φ⟩ = 0 → no CW potential for σ!
#
# WAIT: let's think again.
# V_CW(σ) comes from the FULL propagator with all insertions.
# Even without ⟨φ⟩, at 2-loop there's a contribution:
# χ loop with two φ propagators (sunset diagram).
# But at 1-loop, if ⟨φ⟩ = 0, the only σ-dependent piece is:
#   the χ propagator: m_eff = m_χ (independent of σ if φ=0!)
#
# So: at 1-loop with φ VEV = 0: m_σ = 0 (no CW for σ)
# First contribution at 2-loop: sunset diagram

# Case A: if φ somehow has a VEV (e.g., from μ₃ tadpole or thermal)
# Effective φ VEV from cubic: ⟨φ⟩ ~ μ₃/m_φ × v_loop where v_loop ~ y m_χ/(4π)
# Actually in our model:
# V(φ) = ½ m_φ² φ² + μ₃/6 φ³ + λ₄/24 φ⁴
# Minimum at φ=0 if μ₃ > 0 (and looking at the right vacuum).
# But μ₃ ≠ 0 means there IS a second minimum at φ_min ~ -3μ₃/λ₄
# For cannibal: μ₃/m_φ ≳ 1.7 → μ₃ ~ 2 m_φ
# λ₄ ~ 1 (perturbative)
# φ_min ~ -6 m_φ / 1 ~ -6 m_φ

# Actually, in cosmology, the DM halo fills φ with a classical field.
# In the Born-Oppenheimer picture:
# n_χ of DM sources φ: (∇²-m_φ²)φ = -y_s n_χ → φ_bg = y_s n_χ/m_φ²

# At cosmological density:
rho_crit = 3 * H_0**2 * M_Pl**2 / (8*np.pi)
rho_DM = 0.26 * rho_crit
n_chi_cosmo = rho_DM / m_chi
phi_bg_cosmo = y * np.cos(theta) * n_chi_cosmo / m_phi**2

# In a DM halo (ρ ~ 0.3 GeV/cm³ = 0.3 GeV × (5.07e13)³ /cm³):
rho_halo_SI = 0.3  # GeV/cm³
cm_to_GeV = 5.068e13  # 1/cm in GeV
rho_halo = rho_halo_SI * cm_to_GeV**3  # in GeV⁴
n_chi_halo = rho_halo / m_chi
phi_bg_halo = y * np.cos(theta) * n_chi_halo / m_phi**2

print(f"\n--- Background φ field from DM ---")
print(f"Cosmological: n_χ = {n_chi_cosmo:.2e} GeV³")
print(f"  φ_bg = {phi_bg_cosmo:.2e} GeV")
print(f"  φ_bg / m_φ = {phi_bg_cosmo/m_phi:.2e}")
print(f"\nGalactic halo (ρ=0.3 GeV/cm³): n_χ = {n_chi_halo:.2e} GeV³")
print(f"  φ_bg = {phi_bg_halo:.2e} GeV")
print(f"  φ_bg / m_φ = {phi_bg_halo/m_phi:.2e}")

# φ_bg is TINY compared to m_φ, so the CW potential from this is negligible.
# The CW mass from background DM:
# ΔM² = m_χ y cos(σ/f) φ_bg
#   ≈ m_χ y φ_bg (this is σ/f-dependent piece)
# d²V_CW/dσ² = (y² φ_bg²)/(32π² f²) × m_χ² × [log terms]
# Simplified: m_σ² ~ y² m_χ² φ_bg² / (16π² f²)

# However, this is the WRONG way to think about it.
# The real CW potential comes from virtual φ exchange, not background φ.
# The 2-loop sunset diagram: χ-loop with 2 φ propagators
# V_CW^(2-loop) ~ y⁴ m_χ² / (256 π⁴) × f(m_χ/m_φ)
# This doesn't depend on f — it's a direct loop.

# The σ mass from this 2-loop:
# m_σ² ~ y⁴ m_χ² / (256 π⁴ f²)  (with σ/f factor from vertex)

print(f"\n--- CW mass of σ ---")

# Let me be careful about the loop structure.
# At 1-loop: V_CW depends on σ ONLY through ⟨φ⟩.
# If ⟨φ⟩ = 0 → no 1-loop CW for σ.
# At 2-loop: sunset diagram (2 χ propagators, 1 φ propagator, 
#            or 1 χ loop with 2 φ insertions) gives σ-dependent V.

# For the 2-loop sunset:
# V² ~ (y²/16π²)² × m_χ⁴ × cos(σ/f) × G(m_χ, m_φ)
# where G ~ ln(m_χ/m_φ) ~ ln(94/11) ~ 2.1

# With 2 vertices each carrying cos(σ/f) or sin(σ/f):
# The σ-dependent piece from the sunset goes as cos(2σ/f)
# d²V/dσ² at σ=σ_relic gives:

# Estimate:
# V_sunset ~ (y⁴/(256π⁴)) × m_χ⁴ × cos(2σ/f)
# m_σ² = (4/f²) V_sunset = 4 y⁴ m_χ⁴ / (256 π⁴ f²)

# BUT: We also previously computed V_CW assuming ⟨φ⟩ ≠ 0.
# The papers before used a different picture where φ gets a VEV.
# Let me compute BOTH cases.

print("\nCase A: ⟨φ⟩ = 0 (no tree-level VEV for φ)")
print("  σ mass comes from 2-loop sunset diagram")

for f_val, f_name in [(0.2*M_Pl, "0.2 M_Pl"), 
                       (1e16, "10¹⁶ GeV"),
                       (1e14, "10¹⁴ GeV"),
                       (1e13, "10¹³ GeV")]:
    
    # 2-loop sunset
    V_sunset = y**4 * m_chi**4 / (256 * np.pi**4)
    m_sigma_sq = 4 * V_sunset / f_val**2
    m_sigma = np.sqrt(abs(m_sigma_sq))
    
    print(f"\n  f = {f_name}:")
    print(f"    V_sunset = {V_sunset:.2e} GeV⁴")
    print(f"    m_σ = {m_sigma:.2e} GeV = {m_sigma/eV:.2e} eV")
    print(f"    m_σ / H₀ = {m_sigma/H_0:.2e}")
    print(f"    Range = ℏc/m_σ = {1.97e-14/m_sigma:.2e} cm")

# Case B: if we use the OLD calculation where V_CW assumed φ has a VEV
# V_CW^(1) ~ m_χ⁴/(64π²) × [stuff involving cos(σ/f)]
# This gives the larger estimate from the earlier code

print("\n\nCase B: Effective ⟨φ⟩ from μ₃ tadpole or dense environment")
# If φ has VEV ~ m_phi (from the second minimum of the cubic potential)
phi_vev = m_phi  # conservative estimate

for f_val, f_name in [(0.2*M_Pl, "0.2 M_Pl"), 
                       (1e16, "10¹⁶ GeV"),
                       (1e14, "10¹⁴ GeV"),
                       (1e13, "10¹³ GeV")]:
    
    # 1-loop CW with φ VEV
    # M_eff² = m_χ² + m_χ y cos(θ) φ + y²φ²/4
    # σ-dependent part: m_χ y φ cos(σ/f)
    # d²V_CW/dσ² ~ (m_χ y φ)² / (16π² f²)
    
    Delta = m_chi * y * phi_vev
    m_sigma_sq = Delta**2 / (16 * np.pi**2 * f_val**2)
    m_sigma = np.sqrt(m_sigma_sq)
    
    print(f"\n  f = {f_name}, ⟨φ⟩ = m_φ = {m_phi*1e3:.1f} MeV:")
    print(f"    m_σ = {m_sigma:.2e} GeV = {m_sigma/eV:.2e} eV")
    print(f"    m_σ / H₀ = {m_sigma/H_0:.2e}")

# ============= STEP 2: Protection Mechanisms ==============
print("\n\n" + "="*70)
print("  STEP 2: PROTECTION MECHANISMS")
print("="*70)

# ---------- Mechanism 1: Pure Goldstone ----------
print("\n--- Mechanism 1: EXACT GOLDSTONE (m_σ = 0) ---")
print("""
If the U(1) rotating σ is EXACT: m_σ = 0 by Goldstone theorem.
But U(1): σ→σ+c means y_s, y_p independent. With y_s≠y_p, the U(1) is
EXPLICITLY broken → σ gets mass. Unless:

  The coupling is y e^{iγ⁵σ/f} — this HAS exact shift symmetry!
  σ → σ + 2πf maps to itself.

  But: perturbative expansion breaks this to discrete Z_N.
  Non-perturbatively, it's a cosine potential V ~ Λ⁴(1-cos σ/f).
  
  The mass is: m_σ²  = Λ⁴/f²

  If Λ comes from the fermion determinant:
    Λ⁴ ~ (m_χ y ⟨φ⟩)² / (16π²)  [instanton-like]
    
  This is the SAME as Case B above. No improvement.
  
  BUT: if Λ is from a DIFFERENT scale (e.g., dark QCD):
    Λ_dark ~ few eV → m_σ ~ Λ²/f
""")

# What Λ_dark gives m_σ = H₀?
for f_val, f_name in [(0.2*M_Pl, "0.2 M_Pl"), (1e14, "10¹⁴"), (1e13, "10¹³")]:
    Lambda_needed = np.sqrt(H_0 * f_val)
    print(f"  f = {f_name}: need Λ_dark = {Lambda_needed:.2e} GeV = {Lambda_needed/eV:.2e} eV")

# ---------- Mechanism 2: SUSY cancellation ----------
print("\n--- Mechanism 2: DARK SUSY ---")
print("""
In SUSY, boson and fermion loops cancel. The residual CW mass:
  m_σ² ~ (m_boson² - m_fermion²)² / (16π² f²)
  
In our model: χ (fermion, m_χ) + φ (boson, m_φ)
If dark SUSY: m_χ ≈ m_φ → cancellation!
""")

# SUSY cancellation factor
delta_m_sq = m_chi**2 - m_phi**2
susy_factor = delta_m_sq / m_chi**2  # how much cancellation

print(f"  m_χ = {m_chi*1e3:.1f} MeV, m_φ = {m_phi*1e3:.1f} MeV")
print(f"  m_χ²-m_φ² = {delta_m_sq:.2e} GeV²")
print(f"  Cancellation factor: (m_χ²-m_φ²)/m_χ² = {susy_factor:.4f}")
print(f"  → Almost NO cancellation! m_χ >> m_φ (ratio = {m_chi/m_phi:.1f})")

# What if m_χ ≈ m_φ?
print(f"\n  For SUSY to work, need m_χ ≈ m_φ.")
print(f"  MAP has m_χ/m_φ = {m_chi/m_phi:.1f} → SUSY broken by factor ~{m_chi/m_phi:.0f}")
print(f"  VERDICT: ✗ Dark SUSY doesn't help for our benchmark points.")

# But let's check: what if there's a DIFFERENT BP where m_χ ≈ m_φ?
print(f"\n  Hypothetical: m_χ = 12 MeV, m_φ = 11 MeV (Δm = 1 MeV)")
m_chi_hyp = 12e-3
m_phi_hyp = 11e-3
delta_hyp = m_chi_hyp**2 - m_phi_hyp**2
for f_val, f_name in [(0.2*M_Pl, "0.2 M_Pl"), (1e14, "10¹⁴")]:
    m_sig_susy = np.sqrt(abs(delta_hyp**2 / (16*np.pi**2 * f_val**2)))
    print(f"    f={f_name}: m_σ = {m_sig_susy/eV:.2e} eV, m_σ/H₀ = {m_sig_susy/H_0:.2e}")

# ---------- Mechanism 3: Clockwork ----------
print("\n--- Mechanism 3: CLOCKWORK ---")
print("""
N copies of σ (gears) connected by mass terms.
Only σ_1 couples to the dark sector with strength y.
The lightest mode (σ_0) has coupling suppressed by q^N:
  g_eff = y / q^N

CW mass of physical σ₀:
  m_σ₀² ~ m_σ_CW² / q^{2N}

With q=3 (the A₄ number, cute!):
""")

# Using 2-loop sunset mass as starting point
m_sigma_CW = np.sqrt(4 * y**4 * m_chi**4 / (256*np.pi**4)) / (0.2*M_Pl)

print(f"  Base CW mass (f=0.2 M_Pl): m_σ = {m_sigma_CW/eV:.2e} eV")
print(f"  Target: m_σ = H₀ = {H_0/eV:.2e} eV")
print(f"  Need suppression: {m_sigma_CW/H_0:.2e}")
print(f"  log₃(suppression) = {np.log(m_sigma_CW/H_0)/np.log(3):.1f}")

for q in [2, 3, 5]:
    N_needed = np.log(m_sigma_CW/H_0) / (2*np.log(q))
    print(f"\n  q = {q}: need N = {N_needed:.0f} gears")
    print(f"    Total fields: {N_needed:.0f}+1 copies of σ")
    print(f"    g_eff/y = 1/{q}^{N_needed:.0f} = {q**(-N_needed):.2e}")

# ---------- Mechanism 4: Sequestering / Extra Dimension ----------
print("\n\n--- Mechanism 4: EXTRA DIMENSION ---")
print("""
σ lives on a brane at distance L from the DM brane.
Coupling suppressed by warp factor: g_eff ~ g × e^{-m_* L}
where m_* is the bulk mass scale.

Equivalent to clockwork with continuous N.
Need: e^{-m_* L} ~ H₀/m_σ_CW
""")

# ---------- Mechanism 5: Radiative Seesaw ----------
print("\n--- Mechanism 5: RADIATIVE SEESAW ---")
print("""
Key insight: m_σ² ∝ 1/f². This is because σ appears as σ/f in 
the Lagrangian. Larger f = weaker coupling = smaller mass.

What f gives m_σ = H₀?
""")

# Case A: 2-loop
V_sunset = y**4 * m_chi**4 / (256 * np.pi**4)
f_needed_2loop = np.sqrt(4 * V_sunset) / H_0
print(f"\n  2-loop: f = √(4V_sunset)/H₀ = {f_needed_2loop:.2e} GeV = {f_needed_2loop/M_Pl:.1f} M_Pl")

# Case B: 1-loop with ⟨φ⟩ = m_φ
Delta = m_chi * y * m_phi
f_needed_1loop = Delta / (4 * np.pi * H_0)
print(f"  1-loop: f = Δ/(4π H₀) = {f_needed_1loop:.2e} GeV = {f_needed_1loop/M_Pl:.1f} M_Pl")

print(f"\n  → 2-loop: f ~ {f_needed_2loop/M_Pl:.0e} M_Pl (SUPER-Planckian)")
print(f"  → 1-loop: f ~ {f_needed_1loop/M_Pl:.0e} M_Pl (SUPER-Planckian)")
print(f"  → Trans-Planckian f is problematic (quantum gravity)")

# ============= STEP 3: THE TWIST — Discrete σ ==============
print("\n\n" + "="*70)
print("  STEP 3: THE TWIST — WHAT IF θ ISN'T DYNAMICAL?")
print("="*70)
print("""
We showed (Test 10c, research journal) that θ = arcsin(1/3) is a GROUP
THEORY CONSTANT from A₄. It's NOT a dynamical field.

If θ is discrete → there is NO σ → m_σ is meaningless.

But then: where does DE come from?

The CW vacuum energy at θ_relic is fixed, and it's V_CW ~ 10⁻⁷ GeV⁴.
This is 40 orders too large for DE.

HOWEVER — this is the STANDARD CC problem. Every QFT has it.
What's SPECIAL about our framework:

1. V_CW(θ) = V_CW(0) × [1 - ε cos(2θ)]  where ε << 1
2. At θ=0: pure scalar → maximum |V_CW|
3. At θ_relic: mixed → slightly less |V_CW|

The DIFFERENCE: ΔV = V_CW(θ_relic) - V_CW(0) is TINY:
""")

# Compute ΔV 
# V_CW ~ -m_χ⁴/(64π²) × [3/2 + ln(M²/μ²)]
# The σ-dependent part: ΔM² = m_χ y⟨φ⟩ cosθ
# ΔV ~ m_χ⁴/(64π²) × (y⟨φ⟩/m_χ)² × (cos²θ_relic - 1)

# Using just the ratio of V_CW at different θ:
# V_CW(θ) ∝ M_eff(θ)⁴ ln(M_eff(θ))
# M_eff(θ) = m_χ [1 + (y⟨φ⟩/2m_χ)cosθ]
# ΔV/V ~ 4 (yφ/2m_χ) Δcosθ ≈ 4 × (y m_φ/2m_χ) × (1-cos θ_relic)

eps = y * m_phi / (2*m_chi)
Delta_cos = 1 - np.cos(theta)
DV_frac = 4 * eps * Delta_cos

print(f"  ε = y m_φ/(2m_χ) = {eps:.4f}")
print(f"  1-cos θ_relic = {Delta_cos:.4f}")
print(f"  ΔV/V_CW ≈ {DV_frac:.4f} = {DV_frac*100:.2f}%")

V_CW_0 = m_chi**4 / (64*np.pi**2)
DV_abs = V_CW_0 * DV_frac
rho_L = 2.58e-47

print(f"\n  V_CW(0) ~ {V_CW_0:.2e} GeV⁴")
print(f"  |ΔV| ~ {DV_abs:.2e} GeV⁴")
print(f"  |ΔV|/ρ_Λ = {DV_abs/rho_L:.2e}")
print(f"  Still {np.log10(DV_abs/rho_L):.0f} orders too large.")

# ============= STEP 4: SUMMARY ==============
print("\n\n" + "="*70)
print("  SUMMARY: CAN m_σ BE PROTECTED?")
print("="*70)
print(f"""
MECHANISM          | RESULT                    | VERDICT
-------------------+---------------------------+--------
1. Exact Goldstone | Need Λ_dark ~ 10⁻³ eV    | Possible but unexplained
2. Dark SUSY       | m_χ/m_φ = {m_chi/m_phi:.0f} → no cancel | ✗ FAILS for our BPs
3. Clockwork (q=3) | Need N ~ {np.log(m_sigma_CW/H_0)/(2*np.log(3)):.0f} gears         | Possible but ad hoc
4. Extra dimension | Equivalent to clockwork   | Same
5. Large f         | Need f ~ {f_needed_2loop/M_Pl:.0e} M_Pl       | ✗ Trans-Planckian

MOST PROMISING: Mechanism 1 (Goldstone with tiny Λ)
  If dark sector has a confining gauge group (dark QCD):
  Λ_dark ~ few × 10⁻³ eV → m_σ ~ Λ²/f ~ H₀
  
  This is the QCD axion mechanism!
  In QCD: m_a ~ Λ_QCD²/f_a ~ (200 MeV)²/(10¹² GeV) ~ 10⁻⁵ eV
  For us: m_σ ~ Λ_dark²/f_dark ~ (10⁻³ eV)²/(10¹⁸ GeV) ~ 10⁻³³ eV ✓

  The question: what is this dark QCD? Can it be part of A₄?

ALTERNATIVE: θ is discrete (A₄), no σ exists.
  Then DE is just the CC problem — V_CW contributes ~10⁻⁷ GeV⁴,
  needs standard CC cancellation. No worse than any other theory,
  but no better either.
""")

# ============= BONUS: Dark QCD scenario ==============
print("="*70)
print("  BONUS: DARK QCD SCENARIO")
print("="*70)
print("""
If the dark sector contains:
  - SU(N_d) dark color (confining at Λ_d)
  - χ in fundamental of SU(N_d) (dark quarks = DM)
  - σ = dark pion (pNGB of chiral symmetry breaking)
  - φ = dark ρ meson or scalar glueball

Then:
  - m_σ ~ Λ_d² / f_σ (same as pion mass in QCD)
  - f_σ ~ Λ_d / (4π) if strongly coupled
  - SIDM from φ exchange (as before)
  - σ is ultralight if Λ_d is tiny
""")

# What confinement scale gives the right numbers?
for f_val in [0.2*M_Pl, 1e16, 1e14]:
    Lambda_d = np.sqrt(H_0 * f_val)
    # f_σ ~ Λ_d / (4π) → self-consistent check
    f_sigma_sc = Lambda_d / (4*np.pi)
    m_sigma_sc = Lambda_d**2 / f_val
    print(f"  f = {f_val:.1e} GeV:")
    print(f"    Λ_d = √(H₀ f) = {Lambda_d:.2e} GeV = {Lambda_d/eV:.2e} eV")
    print(f"    m_σ = Λ²/f = {m_sigma_sc/H_0:.2f} H₀")
    # Self-consistency: f should be >> Λ_d
    print(f"    f/Λ_d = {f_val/Lambda_d:.2e} (need >> 1 for weak coupling: {'✓' if f_val/Lambda_d > 100 else '✗'})")
    print()
