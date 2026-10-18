#!/usr/bin/env python3
"""Executable check of the Derivation layer of "The reference state and the latent heat" (chapter 3).

Mirrors the offset shift of ThermoYImpl::reset / ThermoXImpl::reset, the latent-heat sign test of
ThermoOptionsImpl::from_yaml and the Kirchhoff relation between an "ideal" saturation curve and the caloric data,
for chengcli/kintera at the code pin of doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45),
in numpy and sympy. It does not import kintera.

Run: python3 reference_state_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import math
import sys

import sympy as sp

RGAS = 8.31446  # J mol^-1 K^-1, constants::Rgas


def check_C1_energy_offset_shift_leaves_u_invariant():
    """[C1] u/R = u0 + cv (T - Tref) written with the shifted offset u* = u0 - cv Tref is u* + cv T: the same
    function of T, symbolically."""
    u0, cv, Tref, T = sp.symbols("u0 c_v T_ref T")
    before = u0 + cv * (T - Tref)
    after = (u0 - cv * Tref) + cv * T
    ok = sp.simplify(before - after) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} u(T) with the shifted offset: difference {sp.simplify(before - after)}")
    return ok


def check_C2_entropy_offset_shift_leaves_s_invariant():
    """[C2] For a gas, s/R = s0 + (cv + 1) ln(T/Tref) - ln(p/Pref) equals s* + (cv + 1) ln T - ln p with
    s* = s0 - (cv + 1) ln Tref + ln Pref, symbolically; the code clamps Tref and Pref to at least 1 before the
    logarithms (a card with Tref = 0 shifts by ln 1 = 0)."""
    s0, cv, Tref, Pref, T, p = sp.symbols("s0 c_v T_ref P_ref T p", positive=True)
    before = s0 + (cv + 1) * sp.log(T / Tref) - sp.log(p / Pref)
    after = (s0 - ((cv + 1) * sp.log(Tref) - sp.log(Pref))) + (cv + 1) * sp.log(T) - sp.log(p)
    ok = sp.simplify(sp.expand_log(before - after, force=True)) == 0
    print(f"[C2] {'PASS' if ok else 'FAIL'} s(T, p) with the shifted offset: difference "
          f"{sp.simplify(sp.expand_log(before - after, force=True))}; clamp max(Tref, 1) = 1 for Tref = 0")
    return ok


def check_C3_latent_heat_sign_rule():
    """[C3] The load-time test sums u0_R over reactants minus products. For jupiter.yaml's H2O <=> H2O(l) it is
    0 - (-3430) = 3430 K > 0 (silent); for a condensate without u0_R it is 0 (warned, as tests/test_latent_heat_sign.py
    asserts)."""
    latent_jupiter = (1 * 0.0) - (1 * -3430.0)
    latent_bare = (1 * 0.0) - (1 * 0.0)
    ok = latent_jupiter > 0 and not latent_bare > 0
    print(f"[C3] {'PASS' if ok else 'FAIL'} latent heat at Tref: H2O <=> H2O(l) {latent_jupiter:.0f} K (silent); "
          f"bare condensate {latent_bare:.0f} K (warned)")
    return ok


def check_C4_kirchhoff_form_of_the_ideal_curve():
    """[C4] For ln p_sat = ln p_r + beta (1 - T_r/T) - delta ln(T/T_r), Clausius-Clapeyron with an ideal vapour and a
    condensate of negligible volume gives L = R_v T^2 d ln p_sat/dT = R_v (beta T_r - delta T): linear in T with
    slope -R_v delta = c_pv - c_l. With the constant-cv data, (h_v - h_c)/R = (u_v* - u_c*) + (cv_v + 1 - cv_c) T,
    so a consistent card has delta = cv_c - cv_v - 1 and beta T_r = u_v* - u_c*. Checked symbolically."""
    T, Tr, pr, beta, delta = sp.symbols("T T_r p_r beta delta", positive=True)
    lnp = sp.log(pr) + beta * (1 - Tr / T) - delta * sp.log(T / Tr)
    L_over_Rv = sp.simplify(T**2 * sp.diff(lnp, T))
    ok = sp.simplify(L_over_Rv - (beta * Tr - delta * T)) == 0
    print(f"[C4] {'PASS' if ok else 'FAIL'} T^2 d ln p_sat/dT = {L_over_Rv} (Kirchhoff form beta T_r - delta T)")
    return ok


def check_C5_bryan_beta_from_the_benchmark_latent_heat():
    """[C5] h2o_bryan's beta reproduces L_v0 = 2.5e6 J kg^-1 at T_0 = 273.15 K with t_r = 273.16 K and
    delta = 4.986009: beta = (L_v0/R_v + delta T_0)/t_r with R_v = R/M(H2O), M = 2*1.008 + 15.999 g/mol.
    The code's literal is 24.815845; agreement to 1e-7 relative (the literal has 8 significant digits)."""
    M = (2 * 1.008 + 15.999) * 1e-3
    Rv = RGAS / M
    beta = (2.5e6 / Rv + 4.986009 * 273.15) / 273.16
    ok = abs(beta / 24.815845 - 1) < 1e-7
    L = Rv * (beta * 273.16 - 4.986009 * 273.15)
    print(f"[C5] {'PASS' if ok else 'FAIL'} h2o_bryan: beta = {beta:.9f} (code 24.815845), R_v = {Rv:.6f} J kg^-1 K^-1, "
          f"L(273.15 K) = {L:.3f} J kg^-1")
    return ok


def check_C6_deep_jupiter_card_is_kirchhoff_consistent():
    """[C6] examples/jupiter_kcl_mgsio3.yaml: for L = ln(10)(A - B/T) the implied delta h/R is -B ln 10 and
    delta cp = 0. KCl: u0_R(KCl(s)) = -11382 ln 10 and cv_R: 2.5 + 1 - 3.5 = 0. MgSiO3: u0_R = -49895 ln 10 and
    (1.5 + 1) + (8 + 1) + 3 (3 + 1) - 6 - 5 (2.5 + 1) = 0. Tolerance 1e-9 relative on the card's literals."""
    kcl = -11382.0 * math.log(10.0)
    mgsio3 = -49895.0 * math.log(10.0)
    dcp_kcl = (2.5 + 1) - 3.5
    dcp_mg = (1.5 + 1) + (8.0 + 1) + 3 * (3.0 + 1) - 6.0 - 5 * (2.5 + 1)
    ok = (abs(kcl / -26208.02352845823 - 1) < 1e-9 and abs(mgsio3 / -114887.48321493792 - 1) < 1e-9
          and dcp_kcl == 0 and dcp_mg == 0)
    print(f"[C6] {'PASS' if ok else 'FAIL'} card: u0_R(KCl(s)) = {kcl:.8f} K, u0_R(MgSiO3(s)) = {mgsio3:.8f} K, "
          f"delta cp/R = {dcp_kcl}, {dcp_mg}")
    return ok


def main():
    checks = [check_C1_energy_offset_shift_leaves_u_invariant, check_C2_entropy_offset_shift_leaves_s_invariant,
              check_C3_latent_heat_sign_rule, check_C4_kirchhoff_form_of_the_ideal_curve,
              check_C5_bryan_beta_from_the_benchmark_latent_heat, check_C6_deep_jupiter_card_is_kirchhoff_consistent]
    failed = sum(not c() for c in checks)
    print(f"reference_state_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
