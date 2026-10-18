#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Phase-change reactions and the saturation conditions" (chapter 4).

Mirrors the residual, the pressure shift and the active-set test of equilibrate_tp (src/thermo/equilibrate_tp.h)
and the generic Antoine form of src/vapors/vapor_functions.h of chengcli/kintera at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy and sympy. It does not import
kintera.

Run: python3 conditions_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import math
import sys

import numpy as np
import sympy as sp


def check_C1_residual_in_mole_fractions():
    """[C1] With p_i = p x_i / x_g for the gases, r_j = sum_i a_ij ln p_i - L_j equals
    sum_i a_ij ln(x_i/x_g) + A_j ln p - L_j, A_j = sum_i a_ij: the mole-fraction form the TP kernel evaluates
    (log_frac_sum minus logsvp[j] - stoich_sum ln p). Symbolic, two gases with exponents a_1, a_2."""
    x1, x2, xg, p, L, a1, a2 = sp.symbols("x_1 x_2 x_g p L a_1 a_2", positive=True)
    r_pressure = a1 * sp.log(p * x1 / xg) + a2 * sp.log(p * x2 / xg) - L
    r_code = a1 * sp.log(x1 / xg) + a2 * sp.log(x2 / xg) - (L - (a1 + a2) * sp.log(p))
    diff = sp.simplify(sp.expand_log(r_pressure - r_code, force=True))
    ok = diff == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} r_j in partial pressures minus the kernel's mole-fraction form: {diff}")
    return ok


def check_C2_net_gas_moles_of_the_example_reactions():
    """[C2] A_j = -sum_{gas} nu_ij for the chapter's examples: H2O <=> H2O(l) (1), NH3 + H2S <=> NH4SH(s) (2),
    Mn + H2S <=> MnS(s) + H2 (1), Mg + SiH4 + 3 H2O <=> MgSiO3(s) + 5 H2 (0)."""
    cases = {
        "H2O <=> H2O(l)": ([-1.0], 1),
        "NH3 + H2S <=> NH4SH(s)": ([-1.0, -1.0], 2),
        "Mn + H2S <=> MnS(s) + H2": ([-1.0, -1.0, 1.0], 1),
        "Mg + SiH4 + 3 H2O <=> MgSiO3(s) + 5 H2": ([-1.0, -1.0, -3.0, 5.0], 0),
    }
    ok = True
    out = []
    for name, (nu_gas, want) in cases.items():
        A = -sum(nu_gas)
        ok &= A == want
        out.append(f"{name}: A = {A:g}")
    print(f"[C2] {'PASS' if ok else 'FAIL'} " + "; ".join(out))
    return ok


def check_C3_antoine_unit_shift():
    """[C3] The generic Antoine evaluator returns ln(1e5) + (A - B/(T + C)) ln 10: a fit in bar converted to Pa for one
    partial pressure (A_j = 1). For a dimensionless quotient (A_j = 0, the MgSiO3 card) the card subtracts 5 from A so
    that the ln(1e5) cancels: ln(1e5) + (A - 5) ln 10 = A ln 10. Symbolic."""
    A, B, T = sp.symbols("A B T", positive=True)
    antoine = sp.log(sp.Integer(10) ** 5) + (A - 5 - B / T) * sp.log(10)
    want = (A - B / T) * sp.log(10)
    diff = sp.simplify(sp.expand_log(antoine - want, force=True))
    ok = diff == 0
    # the general rule L_Pa = L_bar + A_j ln 1e5, numeric for A_j = 2 (NH4SH: ln Q in Pa^2)
    L_bar = (14.82 - 4705.0 / 200.0) * math.log(10.0) + 2 * math.log(1.01325)  # if the fit were in bar
    L_Pa = L_bar + 2 * math.log(1.0e5)
    ok = ok and abs(L_Pa - ((14.82 - 4705.0 / 200.0) * math.log(10.0) + 2 * math.log(101325.0))) < 1e-9
    print(f"[C3] {'PASS' if ok else 'FAIL'} Antoine with A - 5 for A_j = 0: difference {diff}; "
          f"NH4SH at 200 K: L in Pa^2 = {L_Pa:.4f} = L in bar^2 + 2 ln 1e5")
    return ok


def check_C4_active_set_decision_table():
    """[C4] The kernel activates reaction j when (r_j < -eps and every condensed product is present) or (r_j > +eps):
    supersaturated always condenses; undersaturated evaporates only with a cloud; within +-eps, or undersaturated
    without a cloud, the reaction is in equilibrium. Table of the four cases with eps = 1e-6."""
    eps = 1e-6
    cases = [(+0.5, 0.0, True), (+0.5, 0.1, True), (-0.5, 0.1, True), (-0.5, 0.0, False), (1e-9, 0.1, False)]
    ok = True
    out = []
    for r, x_cloud, want in cases:
        log_frac_sum, logsvp = r, 0.0           # r = log_frac_sum - logsvp
        active = (log_frac_sum < logsvp - eps and x_cloud > 0.0) or (log_frac_sum > logsvp + eps)
        ok &= active == want
        out.append(f"r={r:+.0e}, x_c={x_cloud:g} -> {'active' if active else 'equilibrium'}")
    print(f"[C4] {'PASS' if ok else 'FAIL'} " + "; ".join(out))
    return ok


def check_C5_absent_reactant_linearisation():
    """[C5] With an absent gas reactant (x_i = 0) the kernel replaces ln x_i by the value at the saturation point: the
    stock the reaction would create, r* with nu ln r* = L - (known terms) - nu ln(nu/x_g), then uses the weight
    1/r* (bounded so that exp(-log r*) stays finite) instead of 1/x_i. Checked: for one absent reactant with nu = 1
    the reconstructed ln r* equals L - sum_known - ln(1/x_g), and the bound +-0.5 ln(max double) keeps exp finite."""
    L, known, xg = -2.0, 0.3, 0.98
    nu = 1.0
    log_nu = nu * math.log(abs(nu) / xg)
    log_r = (L - known - log_nu) / nu
    bound = 0.5 * math.log(np.finfo(float).max)
    log_r_b = max(-bound, min(bound, log_r))
    weight = math.exp(-log_r_b)
    ok = abs(log_r - (L - known - math.log(1.0 / xg))) < 1e-14 and math.isfinite(weight)
    # an extremely cold curve: L = -2000 would give exp(2000) without the bound
    log_r_cold = max(-bound, min(bound, (-2000.0 - known - log_nu) / nu))
    ok = ok and math.isfinite(math.exp(-log_r_cold))
    print(f"[C5] {'PASS' if ok else 'FAIL'} absent reactant: ln r* = {log_r:.4f}, weight 1/r* = {weight:.3f}; "
          f"bound {bound:.1f} keeps exp(-ln r*) finite for L = -2000 ({math.exp(-log_r_cold):.3e})")
    return ok


def main():
    checks = [check_C1_residual_in_mole_fractions, check_C2_net_gas_moles_of_the_example_reactions,
              check_C3_antoine_unit_shift, check_C4_active_set_decision_table,
              check_C5_absent_reactant_linearisation]
    failed = sum(not c() for c in checks)
    print(f"conditions_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
