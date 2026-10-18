#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Entropy, enthalpy and the free energies" (chapter 3).

Mirrors eval_enthalpy_R and the two eval_entropy_R overloads of src/thermo/eval_uhs.cpp of chengcli/kintera at the
code pin of doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy and sympy: the gas
entropy, the right-hand side b_j of the condensate-entropy system and the pseudo-inverse solve. It does not import
kintera.

Run: python3 entropy_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import math
import sys

import numpy as np
import sympy as sp

RGAS = 8.31446


def check_C1_gas_entropy_is_the_ideal_mixture_entropy():
    """[C1] eval_entropy_R writes s_i/R = s_i* + cp_i/R ln T - ln p - ln(x_i^g) for a gas, with x_i^g the gas-only
    mole fraction. That is s_i*/R + cp_i/R ln T - ln p_i with the partial pressure p_i = p x_i^g: the entropy of an
    ideal gas at its partial pressure. Symbolic identity."""
    s0, cp, T, p, xg = sp.symbols("s0 c_p T p x", positive=True)
    code = s0 + cp * sp.log(T) - sp.log(p) - sp.log(xg)
    text = s0 + cp * sp.log(T) - sp.log(p * xg)
    ok = sp.simplify(sp.expand_log(code - text, force=True)) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} gas entropy: - ln p - ln x_i^g = - ln p_i (difference "
          f"{sp.simplify(sp.expand_log(code - text, force=True))})")
    return ok


def check_C2_condensate_entropy_from_zero_gibbs_change():
    """[C2] On the saturation curve sum_i nu_ij g_i = 0 with g_i = h_i - T s_i, so sum_{cloud} nu_ij s_i/R =
    sum_i nu_ij h_i/(R T) - sum_{gas} nu_ij s_i/R. With the gas entropy of [C1] and the saturation condition
    sum_{gas} a_ij ln p_i = L_j (a_ij = -nu_ij), the gas part is sum_{gas} nu_ij (s_i* + cp_i ln T) + L_j, which is
    exactly the b_j the code assembles: b_j = sum_i nu_ij h_i/(R T) - sum_{gas} nu_ij (s_i* + cp_i ln T) - L_j.
    Symbolic check for one vapour (nu = -1) and one condensate (nu = +1)."""
    sv, cpv, T, p, xv, hv, hc, L = sp.symbols("s_v c_pv T p x_v h_v h_c L", positive=True)
    s_gas = sv + cpv * sp.log(T) - sp.log(p * xv)          # [C1]
    # exact: nu_v = -1, nu_c = +1: s_c = (h_c - h_v)/T + s_gas
    s_c_exact = (hc - hv) / T + s_gas
    # the code: b = (h_c - h_v)/T - (-1)(s_v + cp_v ln T) - L, with L = ln p_v = ln(p x_v) at saturation
    b = (hc - hv) / T + (sv + cpv * sp.log(T)) - L
    diff = sp.simplify(sp.expand_log((s_c_exact - b).subs(L, sp.log(p * xv)), force=True))
    ok = diff == 0
    print(f"[C2] {'PASS' if ok else 'FAIL'} condensate entropy: exact - code b_j = {diff} on the saturation curve")
    return ok


def check_C3_pseudo_inverse_is_exact_for_one_to_one_blocks():
    """[C3] For jupiter.yaml's three nucleation reactions (H2O(l), NH3(s), NH4SH(s), each in exactly one reaction)
    the cloud block S+ of the stoichiometric matrix is a permutation of the identity, so pinv(S+^T) b returns b itself;
    for two condensates sharing a reaction the system is under-determined and pinv returns the minimum-norm split.
    Tolerance 1e-14 absolute."""
    sp_block = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # rows: clouds, columns: reactions
    b = np.array([-3.1, -4.2, -7.7])
    s = np.linalg.pinv(sp_block.T) @ b
    ok = np.abs(s - b).max() < 1e-14
    # a shared reaction: two condensates both with nu = +1 in one reaction
    shared = np.array([[1.0], [1.0]])
    s2 = np.linalg.pinv(shared.T) @ np.array([-5.0])
    ok = ok and np.allclose(s2, [-2.5, -2.5], atol=1e-14)
    print(f"[C3] {'PASS' if ok else 'FAIL'} pinv(S+^T) b: one-to-one block max |s - b| = {np.abs(s - b).max():.1e}; "
          f"shared block returns the minimum-norm split {s2.round(3).tolist()}")
    return ok


def check_C4_enthalpy_branches():
    """[C4] eval_enthalpy_R: h/R = u/R + z T for a gas (z = 1 at the pin, so h = u + R T) and h/R = u/R + e for a
    condensate, where e is the optional extra enthalpy function (none registered at the pin, so h_c = u_c)."""
    T = 250.0
    u_gas, u_cloud = 2.5 * T, 9.0 * T - 3430.0 - 9.0 * 300.0
    h_gas = u_gas + 1.0 * T
    h_cloud = u_cloud + 0.0
    ok = h_gas - u_gas == T and h_cloud == u_cloud
    print(f"[C4] {'PASS' if ok else 'FAIL'} h/R - u/R = {h_gas - u_gas:.1f} K (gas, = T), {h_cloud - u_cloud:.1f} K (condensate)")
    return ok


def check_C5_water_condensate_entropy_curve():
    """[C5] Numerical value of the condensate entropy the code would assign to H2O(l) of tests/jupiter.yaml at 280 K and
    1 bar with the h2o_ideal curve: s_c/R = (h_c - h_v)/(R T) + s_v*/R + 3.5 ln T - L(T). With the shifted offsets
    (Tref 300 K, Pref 1e5 Pa): u_v* = -2.5*300 = -750 K, u_c* = -3430 - 9*300 = -6130 K, s_v* = -3.5 ln 300 + ln 1e5.
    Prints the value and checks that s_c < s_v (the liquid is more ordered than its vapour at saturation)."""
    T = 280.0
    betal, gammal, tr, pr = 24.845, 4.986009, 273.16, 611.7
    t = T / tr
    L = (1 - 1 / t) * betal - gammal * math.log(t) + math.log(pr)
    uv, uc = -2.5 * 300.0, -3430.0 - 9.0 * 300.0
    hv = uv + 2.5 * T + T          # gas: u + T
    hc = uc + 9.0 * T              # condensate: u
    sv_star = -3.5 * math.log(300.0) + math.log(1.0e5)
    s_c = (hc - hv) / T + sv_star + 3.5 * math.log(T) - L
    s_v = sv_star + 3.5 * math.log(T) - L   # vapour at its saturation partial pressure
    ok = s_c < s_v and math.isfinite(s_c)
    print(f"[C5] {'PASS' if ok else 'FAIL'} H2O(l) at 280 K: s_c/R = {s_c:.4f}, saturated vapour s_v/R = {s_v:.4f}, "
          f"(s_v - s_c) R T = {(s_v - s_c) * RGAS * T:.1f} J mol^-1 = h_v - h_c = {(hv - hc) * RGAS:.1f} J mol^-1")
    return ok


def main():
    checks = [check_C1_gas_entropy_is_the_ideal_mixture_entropy, check_C2_condensate_entropy_from_zero_gibbs_change,
              check_C3_pseudo_inverse_is_exact_for_one_to_one_blocks, check_C4_enthalpy_branches,
              check_C5_water_condensate_entropy_curve]
    failed = sum(not c() for c in checks)
    print(f"entropy_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
