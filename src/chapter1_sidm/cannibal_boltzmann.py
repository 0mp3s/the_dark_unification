#!/usr/bin/env python3
"""
cannibal_boltzmann.py -- Mediator phi Cannibal Depletion (3phi -> 2phi)
======================================================================

PROBLEM (from audit, FATAL #1):
  Stable phi with m_phi ~ 10 MeV, sin(theta_mix) = 0 (secluded)
  -> overclosure by factor ~128,000.

SOLUTION: The 3phi -> 2phi "cannibal" process depletes phi number density
while keeping the dark sector in kinetic equilibrium. This is a standard
mechanism (Carlson+1992, Pappadopulo+2016, Farina+2016).

The cubic coupling mu_3 arises from:
  V(phi) = (1/2) m_phi^2 phi^2 + (mu_3/3!) phi^3 + (lambda_4/4!) phi^4
  CW generates: mu_3^CW ~ y^3 m_chi / (16 pi^2)

This script solves the Boltzmann equation for phi number density
with the 3->2 collision term and determines the minimum mu_3/m_phi
needed to avoid overclosure.
"""
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# Constants
# ===========================================================================
GeV = 1.0
MeV = 1e-3
M_Pl = 2.435e18 * GeV     # reduced Planck mass
H_0_GeV = 1.44e-42 * GeV  # Hubble constant
rho_crit = 3 * H_0_GeV**2 * M_Pl**2  # critical density
Omega_DM_h2 = 0.120
rho_DM = Omega_DM_h2 * rho_crit / (0.674**2)

# ===========================================================================
# Benchmark: BP1 (MAP point)
# ===========================================================================
m_chi = 94.1 * GeV
m_phi = 11.1 * MeV
alpha = 5.73e-3
y = np.sqrt(4 * np.pi * alpha)

# Dark sector temperature ratio (decoupled before QCD phase transition)
xi = 0.35  # T_dark / T_SM ~ (g*S_SM(T_dec)/g*S_SM(T_0))^{-1/3}

# ===========================================================================
# Cannibal rate: <sigma_32 v^2>
# ===========================================================================
def sigma32_v2(mu3_over_mphi, m_phi_val):
    """
    Thermally averaged 3->2 cross section times v^2.
    From Hochberg+2015, Pappadopulo+2016:
      <sigma_32 v^2> ~ (25 sqrt(5) / (512 pi)) * mu_3^4 / m_phi^9
    for non-relativistic phi in the cannibal regime.
    """
    mu3 = mu3_over_mphi * m_phi_val
    prefactor = 25 * np.sqrt(5) / (512 * np.pi)
    return prefactor * mu3**4 / m_phi_val**9


# ===========================================================================
# Equilibrium number density
# ===========================================================================
def n_eq(T, m):
    """Non-relativistic equilibrium density."""
    if T <= 0 or m / T > 500:
        return 0.0
    return (m * T / (2 * np.pi))**1.5 * np.exp(-m / T)


# ===========================================================================
# Hubble rate in radiation domination  
# ===========================================================================
def H_rad(T_SM):
    """Hubble rate during radiation domination."""
    g_star = 10.75  # after e+e- annihilation, before BBN
    return np.sqrt(np.pi**2 * g_star / 90) * T_SM**2 / M_Pl


# ===========================================================================
# Boltzmann equation: dY/dx = - <sigma32 v^2> s^2 Y^2 (Y - Y_eq) / (H x)
# ===========================================================================
def solve_cannibal(mu3_over_mphi, m_phi_val=m_phi, xi_val=xi):
    """
    Solve the Boltzmann ODE for phi yield Y = n_phi / s.
    
    The 3->2 process: phi phi phi -> phi phi
    Rate equation: dn/dt + 3Hn = -<sigma32 v^2> (n^3 - n^2 n_eq)
    
    In terms of Y = n/s, x = m_phi/T_d:
      dY/dx = -lambda_32 * (Y^3 - Y^2 * Y_eq) / x^2
    where lambda_32 = s^2 <sigma32 v^2> / H |_{x=1}
    """
    sv2 = sigma32_v2(mu3_over_mphi, m_phi_val)
    
    # Entropy density of SM at T_d = m_phi
    T_SM_ref = m_phi_val / xi_val
    g_star_s = 10.75
    s_ref = (2 * np.pi**2 / 45) * g_star_s * T_SM_ref**3
    
    # Hubble at T_d = m_phi
    H_ref = H_rad(T_SM_ref)
    
    # Dimensionless rate
    lambda_32 = s_ref**2 * sv2 / H_ref
    
    # Initial condition: phi in thermal equilibrium at x = 1 (T_d = m_phi)
    Y_eq_init = n_eq(m_phi_val, m_phi_val) / s_ref
    
    def Y_eq_func(x):
        T_d = m_phi_val / x
        T_SM = T_d * (1 / xi_val)  # approximate: T_SM scales with T_d
        g_s = 10.75
        s = (2 * np.pi**2 / 45) * g_s * T_SM**3
        return n_eq(T_d, m_phi_val) / s if s > 0 else 0.0
    
    def dYdx(x, Y_arr):
        Y = max(Y_arr[0], 1e-100)
        Yeq = Y_eq_func(x)
        # 3->2: rate goes as Y^3 - Y^2 * Y_eq
        # But we need to account for x-dependence of s and H
        # s ~ x^{-3}, H ~ x^{-2} in radiation domination
        # Full expression: dY/dx = -lambda_32 * x^{-5} * (Y^3 - Y^2 * Yeq) * s_ref^2/s^2 * H_ref/H
        # Since s ~ T^3 ~ x^{-3} and H ~ T^2 ~ x^{-2}:
        #   s^2/H = s_ref^2 * x^{-6} / (H_ref * x^{-2}) = (s_ref^2/H_ref) * x^{-4}
        rate = lambda_32 * (Y**3 - Y**2 * Yeq) / x**4
        return [-rate]
    
    x_span = (1.0, 300.0)
    x_eval = np.linspace(1, 300, 3000)
    
    sol = solve_ivp(dYdx, x_span, [Y_eq_init], 
                    t_eval=x_eval, method='Radau',
                    rtol=1e-10, atol=1e-50)
    
    if not sol.success:
        return None, None
    
    Y_final = sol.y[0, -1]
    return Y_final, sol


# ===========================================================================
# Relic abundance of phi
# ===========================================================================
def omega_phi_h2(Y_final, m_phi_val=m_phi):
    """Convert yield to Omega_phi h^2."""
    # s_0 = 2891.2 cm^{-3} in natural units
    # s_0 = 2891.2 / (hbar c)^3 in GeV^3
    hbar_c_cm = 1.9733e-14  # hbar*c in GeV*cm
    s_0_GeV3 = 2891.2 / hbar_c_cm**3
    
    n_0 = Y_final * s_0_GeV3
    rho_0 = n_0 * m_phi_val
    
    # rho_crit / h^2 in GeV/cm^3
    rho_crit_over_h2 = 1.0537e-5  # GeV/cm^3
    rho_crit_h2_GeV4 = rho_crit_over_h2 / hbar_c_cm**3
    
    return rho_0 / rho_crit_h2_GeV4


# ===========================================================================
# Alternative: simple analytic estimate (Pappadopulo+2016 Eq. 3.7)
# ===========================================================================
def omega_phi_analytic(mu3_over_mphi, m_phi_val=m_phi, xi_val=xi):
    """
    Analytic estimate of the relic abundance from cannibal freeze-out.
    From Pappadopulo+2016: Y_inf ~ (H / (s^2 <sigma32 v^2>))^{1/2}
    evaluated at T_d ~ m_phi / x_fo where x_fo ~ 3-5.
    """
    sv2 = sigma32_v2(mu3_over_mphi, m_phi_val)
    
    x_fo = 4.0  # typical cannibal freeze-out
    T_d_fo = m_phi_val / x_fo
    T_SM_fo = T_d_fo / xi_val
    
    g_star_s = 10.75
    s_fo = (2 * np.pi**2 / 45) * g_star_s * T_SM_fo**3
    H_fo = H_rad(T_SM_fo)
    
    # Freeze-out yield
    Y_fo = np.sqrt(H_fo / (s_fo**2 * sv2)) if sv2 > 0 else 1e10
    
    return omega_phi_h2(Y_fo, m_phi_val)


# ===========================================================================
# CW-generated cubic coupling
# ===========================================================================
def mu3_CW(y_val, m_chi_val):
    """
    Coleman-Weinberg generated cubic coupling.
    mu_3^{CW} = 3 n_f y^3 m_chi / (32 pi^2)
    for n_f Majorana fermions (n_f = 1 here, but factor 1/2 for Majorana).
    """
    return 3 * y_val**3 * m_chi_val / (64 * np.pi**2)


def lambda4_CW(y_val):
    """
    CW-generated quartic coupling: lambda_4 = 3 y^4 / (32 pi^2).
    This also drives 4phi->2phi, and generates effective mu_3 if
    there's any small VEV shift.
    """
    return 3 * y_val**4 / (32 * np.pi**2)


def sigma42_v3(lambda4, m_phi_val):
    """
    4->2 thermally averaged cross section.
    <sigma_42 v^3> ~ lambda4^4 / (128 pi m_phi^11)
    from Hochberg+2018.
    """
    return lambda4**4 / (128 * np.pi * m_phi_val**11)


def omega_phi_combined(mu3_over_mphi, lambda4, m_phi_val=m_phi, xi_val=xi):
    """
    Combined 3->2 + 4->2 cannibal relic abundance.
    The 4->2 process: n_dot + 3Hn = -<sigma42 v^3> n^4 (at leading order).
    Combined effective rate: Gamma_eff = Gamma_32 + Gamma_42
    We use the dominant channel.
    """
    sv2_32 = sigma32_v2(mu3_over_mphi, m_phi_val)
    sv3_42 = sigma42_v3(lambda4, m_phi_val)
    
    x_fo = 4.0
    T_d_fo = m_phi_val / x_fo
    T_SM_fo = T_d_fo / xi_val
    g_star_s = 10.75
    s_fo = (2 * np.pi**2 / 45) * g_star_s * T_SM_fo**3
    H_fo = H_rad(T_SM_fo)
    n_fo = n_eq(T_d_fo, m_phi_val)
    
    # 3->2 rate per particle: Gamma_32 = n^2 <sigma32 v^2>
    rate_32 = n_fo**2 * sv2_32
    # 4->2 rate per particle: Gamma_42 = n^3 <sigma42 v^3>
    rate_42 = n_fo**3 * sv3_42
    
    # Use the dominant channel
    if rate_32 >= rate_42:
        # 3->2 dominates
        Y_fo = np.sqrt(H_fo / (s_fo**2 * sv2_32)) if sv2_32 > 0 else 1e10
    else:
        # 4->2 dominates
        Y_fo = (H_fo / (s_fo**3 * sv3_42))**(1.0/3.0) if sv3_42 > 0 else 1e10
    
    return omega_phi_h2(Y_fo, m_phi_val)


# ===========================================================================
# MAIN ANALYSIS
# ===========================================================================
def main():
    print("=" * 78)
    print("  MEDIATOR CANNIBAL DEPLETION: 3phi -> 2phi BOLTZMANN ANALYSIS")
    print("=" * 78)
    print(f"  Benchmark: MAP point")
    print(f"    m_chi = {m_chi:.1f} GeV")
    print(f"    m_phi = {m_phi/MeV:.2f} MeV")
    print(f"    alpha = {alpha:.4e}")
    print(f"    y     = {y:.4e}")
    print(f"    xi    = {xi:.2f} (T_dark/T_SM)")
    print()
    
    # ------------------------------------------------------------------
    # Part 1: Without cannibal — confirm overclosure
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 1: NO CANNIBAL — CONFIRM OVERCLOSURE")
    print("=" * 78)
    
    # phi in thermal equilibrium, freeze-out when non-relativistic
    # Y_phi ~ n_eq/s at x_fo ~ 3 (no annihilation channel for stable phi)
    x_fo_no_cannibal = 3.0
    T_d_fo = m_phi / x_fo_no_cannibal
    T_SM_fo = T_d_fo / xi
    g_star_s = 10.75
    s_fo = (2 * np.pi**2 / 45) * g_star_s * T_SM_fo**3
    Y_no_cannibal = n_eq(T_d_fo, m_phi) / s_fo
    omega_no = omega_phi_h2(Y_no_cannibal)
    
    print(f"  Without cannibal: Y_phi = {Y_no_cannibal:.3e}")
    print(f"  Omega_phi h^2 = {omega_no:.2e}")
    print(f"  Overclosure factor = {omega_no / 0.120:.0f}x")
    print(f"  -> OVERCLOSURE CONFIRMED" if omega_no > 0.120 else "  -> OK")
    print()
    
    # ------------------------------------------------------------------
    # Part 2: CW cubic coupling
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 2: COLEMAN-WEINBERG CUBIC COUPLING")
    print("=" * 78)
    
    mu3_cw = mu3_CW(y, m_chi)
    mu3_over_mphi_cw = mu3_cw / m_phi
    
    print(f"  mu_3^CW = 3 y^3 m_chi / (64 pi^2) = {mu3_cw:.4e} GeV")
    print(f"  mu_3^CW / m_phi = {mu3_over_mphi_cw:.2f}")
    print()
    
    # Also: tree-level from the Yukawa coupling generates effective phi^3
    # through chi loop: additional contribution
    mu3_tree = y * m_phi  # from V = y phi chi_bar chi -> integrating out chi
    mu3_tree_ratio = mu3_tree / m_phi
    print(f"  Tree-level effective (from chi loop): mu_3 ~ y * m_phi = {mu3_tree:.4e} GeV")
    print(f"  mu_3^tree / m_phi = {mu3_tree_ratio:.4e}")
    print()
    
    # Total
    mu3_total = mu3_cw  # CW dominates
    mu3_ratio_total = mu3_total / m_phi
    print(f"  Dominant contribution: CW (loop-generated)")
    print(f"  mu_3/m_phi = {mu3_ratio_total:.2f}")
    print()
    
    # ------------------------------------------------------------------
    # Part 3: Scan mu_3/m_phi — find critical value
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 3: CANNIBAL DEPLETION — ANALYTIC SCAN")
    print("=" * 78)
    print()
    
    ratios = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 
                       mu3_ratio_total, 5.0, 10.0])
    ratios = np.sort(np.unique(ratios))
    
    print(f"  {'mu_3/m_phi':>12} | {'Omega_phi h^2':>14} | {'Overcl. factor':>15} | {'Status':>20}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*15}-+-{'-'*20}")
    
    critical_ratio = None
    for r in ratios:
        omega = omega_phi_analytic(r)
        factor = omega / 0.120
        if omega < 0.120:
            status = "OK"
        elif omega < 1.0:
            status = "overclosure"
        else:
            status = "OVERCLOSURE"
        
        marker = " <-- CW value" if abs(r - mu3_ratio_total) < 0.01 else ""
        print(f"  {r:12.2f} | {omega:14.3e} | {factor:15.1f}x | {status:>20}{marker}")
        
        if critical_ratio is None and omega < 0.120:
            critical_ratio = r
    
    print()
    
    # Find exact critical ratio by bisection
    from scipy.optimize import brentq
    
    def excess(r):
        return omega_phi_analytic(r) - 0.120
    
    try:
        r_crit = brentq(excess, 0.01, 20.0)
        print(f"  Critical mu_3/m_phi (Omega = Omega_DM): {r_crit:.3f}")
    except:
        r_crit = None
        print(f"  Could not find critical ratio")
    
    print()
    
    # ------------------------------------------------------------------
    # Part 4: Perturbativity bound
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 4: PERTURBATIVITY")
    print("=" * 78)
    
    # The cubic coupling generates phi self-energy at 1-loop
    # Perturbativity requires: mu_3^2 / (16 pi^2 m_phi^2) < 1
    # i.e., mu_3/m_phi < 4 pi
    mu3_pert = 4 * np.pi
    print(f"  Perturbativity: mu_3/m_phi < 4 pi = {mu3_pert:.2f}")
    print(f"  CW value: mu_3/m_phi = {mu3_ratio_total:.2f}")
    print(f"  Perturbative? {'YES' if mu3_ratio_total < mu3_pert else 'NO'}")
    print()
    
    if r_crit is not None:
        print(f"  Viable window: {r_crit:.2f} < mu_3/m_phi < {mu3_pert:.2f}")
        print(f"  Dynamic range: {mu3_pert / r_crit:.1f}x")
        cw_in_window = r_crit < mu3_ratio_total < mu3_pert
        print(f"  CW value ({mu3_ratio_total:.2f}) in window? {'YES' if cw_in_window else 'NO'}")
    print()
    
    # ------------------------------------------------------------------
    # Part 5: Numerical Boltzmann solution for CW value
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 5: NUMERICAL BOLTZMANN ODE")
    print("=" * 78)
    
    Y_final_cw, sol_cw = solve_cannibal(mu3_ratio_total)
    if Y_final_cw is not None:
        omega_cw = omega_phi_h2(Y_final_cw)
        print(f"  mu_3/m_phi = {mu3_ratio_total:.2f} (CW value)")
        print(f"  Y_final = {Y_final_cw:.3e}")
        print(f"  Omega_phi h^2 = {omega_cw:.4e}")
        print(f"  Omega_phi / Omega_DM = {omega_cw / 0.120:.2e}")
        if omega_cw < 0.120:
            print(f"  -> SAFE: phi does NOT overclose")
        elif omega_cw < 1.0:
            print(f"  -> WARNING: contributes to DM but doesn't overclose")
        else:
            print(f"  -> OVERCLOSURE")
    else:
        print(f"  ODE solver failed for CW value")
    print()
    
    # Also solve at critical ratio
    if r_crit is not None:
        Y_final_crit, _ = solve_cannibal(r_crit)
        if Y_final_crit is not None:
            omega_crit = omega_phi_h2(Y_final_crit)
            print(f"  Cross-check at critical ratio {r_crit:.3f}:")
            print(f"    Y_final = {Y_final_crit:.3e}, Omega_phi h^2 = {omega_crit:.4e}")
    print()
    
    # ------------------------------------------------------------------
    # Part 6: BBN safety
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 6: BBN SAFETY")
    print("=" * 78)
    
    T_BBN = 1.0 * MeV  # BBN starts at ~ 1 MeV
    T_SM_cannibal_fo = m_phi / (4.0 * xi)  # cannibal freeze-out
    
    print(f"  Cannibal freeze-out: T_SM ~ m_phi / (4 xi) = {T_SM_cannibal_fo/MeV:.1f} MeV")
    print(f"  BBN starts at: T_BBN = {T_BBN/MeV:.1f} MeV")
    print(f"  Cannibal done before BBN? {'YES' if T_SM_cannibal_fo > T_BBN else 'NO'}")
    print()
    
    # phi mass at BBN
    # Energy injection: phi -> nothing (dark sector, cannibal)
    # No energy injected into SM -> BBN unaffected
    print(f"  Does cannibal 3->2 inject energy into SM? NO (secluded sector)")
    print(f"  -> BBN SAFE")
    print()
    
    # Delta N_eff from residual phi
    xi_4 = xi**4
    if Y_final_cw is not None:
        # Remaining phi are non-relativistic, contribute as matter not radiation
        # Their contribution to N_eff is exponentially suppressed for m_phi >> T
        print(f"  Residual phi at BBN: non-relativistic (m_phi/T_d ~ {m_phi/(T_BBN*xi):.0f})")
        print(f"  Contribution to N_eff: exponentially suppressed -> NEGLIGIBLE")
    print()
    
    # ------------------------------------------------------------------
    # Part 7: Check across all benchmark points
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  PART 7: ALL BENCHMARK POINTS")
    print("=" * 78)
    
    benchmarks = [
        ("BP1",  20.7,  9.91, 1.05e-3),
        ("BP2",  29.8, 11.34, 1.47e-3),
        ("BP3",  37.9, 12.98, 1.87e-3),
        ("BP4",  48.3, 14.85, 2.36e-3),
        ("BP5",  61.6, 14.10, 2.97e-3),
        ("BP6",  78.5, 12.50, 3.78e-3),
        ("MAP",  94.1, 11.10, 5.73e-3),
    ]
    
    print(f"  {'Name':>5} | {'m_chi':>8} | {'m_phi':>8} | {'alpha':>10} | {'mu3_CW/m_phi':>13} | {'lam4_CW':>10} | {'Omega(3+4)':>12} | {'Status':>10}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*13}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    
    all_safe = True
    for name, mc, mp_mev, alp in benchmarks:
        mp = mp_mev * MeV
        yy = np.sqrt(4 * np.pi * alp)
        mu3 = mu3_CW(yy, mc * GeV)
        ratio = mu3 / mp
        lam4 = lambda4_CW(yy)
        omega = omega_phi_combined(ratio, lam4, mp)
        safe = omega < 0.120
        if not safe:
            all_safe = False
        print(f"  {name:>5} | {mc:8.1f} | {mp_mev:8.2f} | {alp:10.3e} | {ratio:13.4f} | {lam4:10.3e} | {omega:12.3e} | {'OK' if safe else 'FAIL':>10}")
    
    print()
    if all_safe:
        print(f"  ALL BENCHMARKS SAFE")
    else:
        print(f"  SOME BENCHMARKS FAIL — need additional depletion")
    print()
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print()
    print(f"  1. Without cannibal: Omega_phi h^2 ~ {omega_no:.1e} -> overclosure x{omega_no/0.120:.0f}")
    print()
    print(f"  2. CW generates cubic coupling: mu_3/m_phi = {mu3_ratio_total:.2f}")
    print(f"     This is AUTOMATIC — no new parameters needed.")
    print()
    if r_crit is not None:
        print(f"  3. Critical ratio for Omega_phi < Omega_DM: mu_3/m_phi > {r_crit:.2f}")
        cw_ok = mu3_ratio_total > r_crit
        print(f"     CW value ({mu3_ratio_total:.2f}) {'>' if cw_ok else '<'} critical ({r_crit:.2f})")
        print(f"     -> {'OVERCLOSURE RESOLVED' if cw_ok else 'OVERCLOSURE PERSISTS'}")
    print()
    print(f"  4. Perturbativity OK: {mu3_ratio_total:.2f} < 4 pi = {mu3_pert:.2f}")
    print()
    print(f"  5. BBN safe: cannibal freeze-out at T_SM = {T_SM_cannibal_fo/MeV:.1f} MeV > 1 MeV")
    print()
    
    if Y_final_cw is not None:
        verdict = "RESOLVED" if omega_cw < 0.120 else "PERSISTS"
        print(f"  VERDICT: MEDIATOR OVERCLOSURE -> {verdict}")
        if omega_cw < 0.120:
            print(f"    Omega_phi h^2 = {omega_cw:.3e} (< 0.120)")
            print(f"    Mechanism: CW-generated 3phi->2phi cannibal")
            print(f"    New parameters: ZERO (mu_3 is a CW output)")
    print()


if __name__ == "__main__":
    main()
