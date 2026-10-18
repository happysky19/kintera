#!/usr/bin/env python3
"""Executable check of the Derivation layer of "The saturation ratio" (chapter 3).

Mirrors relative_humidity of src/thermo/relative_humidity.cpp of chengcli/kintera at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy and sympy. It does not import
kintera.

Run: python3 relative_humidity_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import math
import sys

import numpy as np
import sympy as sp

RGAS = 8.31446


def check_C1_quotient_in_concentration_units():
    """[C1] The code evaluates prod_i c_i^{a_ij} / exp(L_j - A_j ln(R T)) with a_ij = -nu_ij over the gases and
    A_j = sum_i a_ij. Since p_i = c_i R T, this equals prod_i p_i^{a_ij} / exp(L_j) = Q_j / Q_j^sat. Symbolic check
    for the NH4SH reaction (a = 1 for NH3 and H2S, A = 2)."""
    c1, c2, T, L = sp.symbols("c_1 c_2 T L", positive=True)
    R = sp.Symbol("R", positive=True)
    code = c1 * c2 / sp.exp(L - 2 * sp.log(R * T))
    text = (c1 * R * T) * (c2 * R * T) / sp.exp(L)
    ok = sp.simplify(code - text) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} NH4SH: c-form minus p-form of Q/Q_sat simplifies to {sp.simplify(code - text)}")
    return ok


def check_C2_reactant_mask_without_and_with_ngas():
    """[C2] Without ngas the exponent matrix is |min(nu, 0)|: every reactant, which is right only when all products
    are condensates. With ngas it is -nu over the first ngas rows: a gaseous product (nu > 0) then enters with a
    negative exponent. Mn + H2S -> MnS(s) + H2: nu = (-1, -1, +1, +1) for (Mn, H2S, H2, MnS(s)), ngas = 3."""
    nu = np.array([-1.0, -1.0, 1.0, 1.0])
    legacy = np.abs(np.minimum(nu, 0.0))
    explicit = np.zeros_like(nu)
    explicit[:3] = -nu[:3]
    ok = legacy.tolist() == [1, 1, 0, 0] and explicit.tolist() == [1, 1, -1, 0]
    print(f"[C2] {'PASS' if ok else 'FAIL'} exponents: legacy {legacy.tolist()} (H2 ignored), with ngas "
          f"{explicit.tolist()} (H2 divides)")
    return ok


def check_C3_water_saturation_ratio_value():
    """[C3] Water vapour at 1 mol m^-3 and 300 K against h2o_ideal: p_v = R T = 2494.3 Pa, p_sat(300 K) = exp(L),
    ratio = p_v/p_sat; the ratio is below 1 (unsaturated) and equals R T / p_sat to 1e-14 relative."""
    T = 300.0
    betal, gammal, tr, pr = 24.845, 4.986009, 273.16, 611.7
    L = (1 - tr / T) * betal - gammal * math.log(T / tr) + math.log(pr)
    rh_code = (1.0 ** 1) / math.exp(L - 1 * math.log(RGAS * T))
    rh_text = RGAS * T / math.exp(L)
    ok = abs(rh_code / rh_text - 1) < 1e-14 and rh_code < 1
    print(f"[C3] {'PASS' if ok else 'FAIL'} H2O at 1 mol m^-3, 300 K: p_sat = {math.exp(L):.1f} Pa, "
          f"saturation ratio = {rh_code:.5f}")
    return ok


def check_C4_zero_amounts():
    """[C4] A zero reactant gives ratio 0; a zero gaseous product (exponent -1) gives +inf; both at once give nan
    (0 * inf), which is why the solvers treat such a reaction as blocked rather than evaluating this quotient."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r0 = np.float64(0.0) ** 1 * np.float64(2.0) ** -1
        rinf = np.float64(2.0) ** 1 * np.float64(0.0) ** -1
        rnan = np.float64(0.0) ** 1 * np.float64(0.0) ** -1
    ok = r0 == 0 and np.isinf(rinf) and np.isnan(rnan)
    print(f"[C4] {'PASS' if ok else 'FAIL'} zero reactant -> {r0}, zero gas product -> {rinf}, both -> {rnan}")
    return ok


def main():
    checks = [check_C1_quotient_in_concentration_units, check_C2_reactant_mask_without_and_with_ngas,
              check_C3_water_saturation_ratio_value, check_C4_zero_amounts]
    failed = sum(not c() for c in checks)
    print(f"relative_humidity_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
