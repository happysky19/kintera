#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Moist adiabatic extrapolation and the effective heat capacity"
(chapter 3).

Mirrors ThermoXImpl::effective_cp and the temperature update of ThermoXImpl::extrapolate_dlnp of
src/thermo/extrapolate_ad.cpp of chengcli/kintera at the code pin of doc/tech-report/code_pin.yml
(0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy and sympy, for one condensing vapour in a dry carrier. It does
not import kintera.

Run: python3 adiabat_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import math
import sys

import numpy as np
import sympy as sp

RGAS = 8.31446


def check_C1_single_reaction_gain():
    """[C1] For V <=> C in a carrier D (nu = (0, -1, +1)), the saturation residual is r = ln(x_v/x_g) + ln p - L(T) and
    the gain G = dr/dxi = sum_i (dr/dx_i) nu_i with dr/dx_i = a_i/x_i - A/x_g (A = 1) is G = 1/x_g - 1/x_v < 0.
    Symbolic."""
    xd, xv, xc, p, L = sp.symbols("x_d x_v x_c p L", positive=True)
    xg = xd + xv
    r = sp.log(xv / xg) + sp.log(p) - L
    G = sp.diff(r, xd) * 0 + sp.diff(r, xv) * (-1) + 0
    ok = sp.simplify(G - (1 / xg - 1 / xv)) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} single-reaction gain G = {sp.simplify(G)} = 1/x_g - 1/x_v")
    return ok


def check_C2_latent_term_closed_form():
    """[C2] effective_cp solves G dxi/dT = L'(T) and adds c_p,latent = (sum_i h_i nu_i) dxi/dT. With
    h_c - h_v = -l (l > 0 the molar latent heat) and G from [C1]: c_p,latent = l L' x_v x_g / (x_g - x_v). For an
    ideal vapour, L' = l/(R T^2), so c_p,latent -> x_v l^2/(R T^2) as x_v/x_g -> 0: the classical moist term.
    Symbolic."""
    xg, xv, l, R, T, Lp = sp.symbols("x_g x_v l R T L_p", positive=True)
    G = 1 / xg - 1 / xv
    dxidT = Lp / G
    cp_latent = (-l) * dxidT
    closed = l * Lp * xv * xg / (xg - xv)
    ok = sp.simplify(cp_latent - closed) == 0
    limit = sp.limit((cp_latent.subs(Lp, l / (R * T**2)) / xv).subs(xg, 1), xv, 0)
    ok = ok and sp.simplify(limit - l**2 / (R * T**2)) == 0
    print(f"[C2] {'PASS' if ok else 'FAIL'} c_p,latent = {closed}; dilute limit per x_v = {limit}")
    return ok


def check_C3_water_effective_cp_value():
    """[C3] A saturated water parcel in dry air at 1 bar and 290 K with the h2o_ideal curve and jupiter.yaml's constant
    heat capacities: c_p,normal = R sum x_i cp_i/R and the latent term of [C2]. Prints both in J mol^-1 K^-1 and
    checks that the latent term is positive and smaller than 10 times the dry value (a sanity bound, not a
    reference value)."""
    T, p = 290.0, 1.0e5
    betal, gammal, tr, pr = 24.845, 4.986009, 273.16, 611.7
    L = (1 - tr / T) * betal - gammal * math.log(T / tr) + math.log(pr)
    Lp = (betal * tr / T**2 - gammal / T)
    xv = math.exp(L) / p
    xg = 1.0
    xd = xg - xv
    cp_normal = RGAS * (xd * 3.5 + xv * 3.5)
    # l = h_v - h_c at T, with the shifted offsets of jupiter.yaml: u_v* = -750 K, u_c* = -6130 K
    hv = (-750.0 + 2.5 * T + T) * RGAS
    hc = (-6130.0 + 9.0 * T) * RGAS
    l = hv - hc
    cp_latent = l * Lp * xv * xg / (xg - xv)
    ok = cp_latent > 0 and cp_latent < 10 * cp_normal
    print(f"[C3] {'PASS' if ok else 'FAIL'} 290 K, 1 bar: x_v = {xv:.5f}, l = {l:.0f} J mol^-1, "
          f"c_p,normal = {cp_normal:.3f}, c_p,latent = {cp_latent:.3f} J mol^-1 K^-1 "
          f"(ratio {cp_latent / cp_normal:.3f})")
    return ok


def check_C4_entropy_newton_step():
    """[C4] extrapolate_dlnp updates T <- T (1 + (s_target - s)/c_p): the Newton step on s(T) - s_target = 0 with
    ds/dT = c_p/T at fixed pressure and composition, linearised in T rather than in ln T. For an ideal dry gas
    s = cp ln T - R ln p the exact update would be T exp(delta) with delta = (s_target - s)/c_p; the code's T (1 + delta)
    leaves a remainder c_p delta^2/2 in the entropy: quadratic convergence with that constant. Checked from a 1 %
    perturbed guess: the remaining error equals e0^2/(2 c_p) to 2 %."""
    cp = 3.5 * RGAS
    T_true, p0, p1 = 300.0, 1.0e5, 0.9e5
    s0 = cp * math.log(T_true) - RGAS * math.log(p0)
    T_exact = T_true * (p1 / p0) ** (RGAS / cp)
    T = 1.01 * T_exact
    s = cp * math.log(T) - RGAS * math.log(p1)
    e0 = abs(s0 - s)
    T = T * (1 + (s0 - s) / cp)
    e1 = abs(s0 - (cp * math.log(T) - RGAS * math.log(p1)))
    predicted = e0**2 / (2 * cp)
    ok = abs(e1 / predicted - 1) < 0.02
    print(f"[C4] {'PASS' if ok else 'FAIL'} entropy Newton: |ds| {e0:.3e} -> {e1:.3e} J mol^-1 K^-1 after one step "
          f"(predicted remainder e0^2/(2 c_p) = {predicted:.3e}); T = {T:.6f} K (exact dry adiabat {T_exact:.6f} K)")
    return ok


def check_C5_rainout_entropy_target():
    """[C5] With rainout the target per mole of gas is (x_dry,new/x_dry,0) s_0 - sum_j s_c,j c_c,j / n_g: for no
    condensation (x_dry unchanged, no cloud) it reduces to s_0; removing cloud of entropy s_c lowers the target by
    the cloud's share. Numeric identity check."""
    s0 = 150.0
    x_dry0, x_dry_new = 0.95, 0.95
    target = (x_dry_new / x_dry0) * s0 - 0.0
    ok = target == s0
    cloud = 0.002 * 60.0 / 1.0  # c_c s_c / n_g
    target2 = (0.96 / 0.95) * s0 - cloud
    ok = ok and target2 < (0.96 / 0.95) * s0
    print(f"[C5] {'PASS' if ok else 'FAIL'} rain-out target: no cloud -> {target} (= s_0); with cloud {target2:.3f} "
          f"(lowered by {cloud:.3f})")
    return ok


def main():
    checks = [check_C1_single_reaction_gain, check_C2_latent_term_closed_form, check_C3_water_effective_cp_value,
              check_C4_entropy_newton_step, check_C5_rainout_entropy_target]
    failed = sum(not c() for c in checks)
    print(f"adiabat_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
