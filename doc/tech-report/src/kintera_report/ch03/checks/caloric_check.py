#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Heat capacities and internal energy" (chapter 3).

Mirrors eval_cv_R, eval_cp_R, eval_intEng_R and the two opt-in models of src/thermo/eval_uhs.cpp (the NASA-9
polynomial and the rigid-rotor H2 partition function) of chengcli/kintera at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy and sympy. The NASA-9
coefficients of H2 are read from ../data/nasa9_h2.dat, a verbatim copy of the pin's data/nasa9.dat record.
It does not import kintera.

Run: python3 caloric_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

T0 = 300.0        # K, kNasa9Tref: the temperature at which the opt-in models join the constant-cv baseline
THETA_ROT = 87.55  # K, kThetaRot
JMAX = 40          # kJmax
DATA = Path(__file__).resolve().parent.parent / "data" / "nasa9_h2.dat"


def nasa9_coefficients():
    lines = [l for l in DATA.read_text().splitlines() if l and not l.startswith("#")]
    v = np.array(" ".join(lines[1:5]).split(), dtype=float)
    return v[0:10], v[10:20]  # low (200-1000 K), high (1000-6000 K): a0..a6, a7=0, a8, a9


def nasa9_cp_R(a, T):
    return a[0] / T**2 + a[1] / T + a[2] + a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4


def nasa9_h_R(a, T):
    """h/R as eval_uhs.cpp writes it: the integration constant is the stored a7 slot, index 7 (b1 of the record)."""
    return (-a[0] / T + a[1] * np.log(T) + a[2] * T + a[3] * T**2 / 2 + a[4] * T**3 / 3 + a[5] * T**4 / 4
            + a[6] * T**5 / 5 + a[7])


def check_C1_nasa9_enthalpy_is_the_integral_of_cp():
    """[C1] d(h/R)/dT = cp/R for the NASA-9 forms of eval_uhs.cpp, symbolically."""
    T = sp.symbols("T", positive=True)
    a = sp.symbols("a0:8")
    cp = a[0] / T**2 + a[1] / T + a[2] + a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4
    h = (-a[0] / T + a[1] * sp.log(T) + a[2] * T + a[3] * T**2 / 2 + a[4] * T**3 / 3 + a[5] * T**4 / 4
         + a[6] * T**5 / 5 + a[7])
    ok = sp.simplify(sp.diff(h, T) - cp) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} NASA-9: d(h/R)/dT - cp/R simplifies to {sp.simplify(sp.diff(h, T) - cp)}")
    return ok


def check_C2_nasa9_internal_energy_joins_the_baseline_at_T0():
    """[C2] The NASA-9 internal energy of eval_intEng_R is u/R = u*/R + T0 cv*/R + [h(T) - h(T0)]/R - (T - T0): it
    equals the constant-cv value u*/R + cv*/R T0 at T = T0 (continuity), and its derivative is cp/R - 1 = cv/R
    (finite difference, tolerance 1e-7 relative on the H2 record at 650 K and 3000 K)."""
    low, high = nasa9_coefficients()
    cref, uref = 2.5, 0.0
    a0 = low if T0 < 1000.0 else high

    def e_R(T):
        a = low if T < 1000.0 else high
        return (nasa9_h_R(a, T) - nasa9_h_R(a0, T0)) - (T - T0)

    def u_R(T):
        return uref + T0 * cref + e_R(T)

    ok = abs(u_R(T0) - (uref + cref * T0)) < 1e-12
    worst = 0.0
    for T in (650.0, 3000.0):
        a = low if T < 1000.0 else high
        fd = (u_R(T * (1 + 1e-6)) - u_R(T * (1 - 1e-6))) / (2e-6 * T)
        cv = nasa9_cp_R(a, T) - 1.0
        worst = max(worst, abs(fd / cv - 1))
    ok = ok and worst < 1e-7
    print(f"[C2] {'PASS' if ok else 'FAIL'} NASA-9 u/R(T0) - baseline = {u_R(T0) - (uref + cref * T0):.1e}; "
          f"max |dU/dT / (cp/R - 1) - 1| = {worst:.1e} (tolerance 1e-7); cp/R(H2, 300 K) = {nasa9_cp_R(low, 300.0):.4f}")
    return ok


def h2_rot(T, mode):
    """<m> and cv_rot/R of the rigid rotor summed to J = JMAX, as h2_rot_reduce/eval_h2cp do."""
    J = np.arange(0, JMAX + 1, dtype=float)
    m = J * (J + 1)
    deg = 2 * J + 1
    odd = (J % 2) == 1

    def reduce(w):
        wx = w * np.exp(-THETA_ROT * m / T)
        Z = wx.sum()
        m1 = (wx * m).sum() / Z
        m2 = (wx * m * m).sum() / Z
        return m1, (THETA_ROT / T) ** 2 * (m2 - m1 * m1)

    if mode == "normal":
        p = reduce(deg * (~odd))
        o = reduce(deg * odd)
        return 0.25 * p[0] + 0.75 * o[0], 0.25 * p[1] + 0.75 * o[1]
    return reduce(deg * np.where(odd, 3.0, 1.0))


def check_C3_h2_rotational_heat_capacity_values():
    """[C3] cp/R = 5/2 + cv_rot/R of the H2 model at the temperatures the test pins: equilibrium 50 K 4.5684 and
    300 K 3.4526; normal 20 K 2.5000, 50 K 2.5038, 300 K 3.4522 (tests/test_h2_cp.py, tolerance 2e-3)."""
    want = {("equilibrium", 50.0): 4.5684, ("equilibrium", 300.0): 3.4526, ("normal", 20.0): 2.5000,
            ("normal", 50.0): 2.5038, ("normal", 300.0): 3.4522}
    got = {k: 2.5 + h2_rot(k[1], k[0])[1] for k in want}
    worst = max(abs(got[k] - want[k]) for k in want)
    ok = worst < 2e-3
    print(f"[C3] {'PASS' if ok else 'FAIL'} H2 cp/R: " + ", ".join(f"{k[0]} {k[1]:.0f} K {got[k]:.4f}" for k in want)
          + f"; max |got - pinned| = {worst:.1e} (tolerance 2e-3)")
    return ok


def check_C4_h2_internal_energy_is_consistent_with_its_cv():
    """[C4] The H2 energy e/R = 3/2 (T - T0) + theta (<m>(T) - <m>(T0)) has derivative 3/2 + cv_rot/R = cv/R
    (finite difference, tolerance 1e-6 relative at 50 K and 300 K, equilibrium mode), and the classical limit
    cp/R -> 7/2 is approached at 1000 K within 1e-3."""
    worst = 0.0
    for T in (50.0, 300.0):
        def e(T):
            return 1.5 * (T - T0) + THETA_ROT * (h2_rot(T, "equilibrium")[0] - h2_rot(T0, "equilibrium")[0])
        fd = (e(T * (1 + 1e-6)) - e(T * (1 - 1e-6))) / (2e-6 * T)
        cv = 1.5 + h2_rot(T, "equilibrium")[1]
        worst = max(worst, abs(fd / cv - 1))
    classical = 2.5 + h2_rot(1000.0, "equilibrium")[1]
    ok = worst < 1e-6 and abs(classical - 3.5) < 1e-3
    print(f"[C4] {'PASS' if ok else 'FAIL'} H2: max |de/dT / cv - 1| = {worst:.1e} (tolerance 1e-6); "
          f"cp/R(1000 K) = {classical:.5f} (classical 3.5)")
    return ok


def check_C5_mayer_relation_in_the_baseline():
    """[C5] In the constant-cv baseline eval_cp_R adds exactly 1 to cref_R of every gas and nothing to a condensate:
    cp/R - cv/R = 1 (gas), 0 (condensate)."""
    cref = np.array([2.5, 2.5, 9.0])  # dry, H2O, H2O(l) of tests/jupiter.yaml
    ngas = 2
    cp = cref.copy()
    cp[:ngas] += 1
    diff = cp - cref
    ok = np.array_equal(diff, [1.0, 1.0, 0.0])
    print(f"[C5] {'PASS' if ok else 'FAIL'} cp/R - cv/R = {diff.tolist()} for (dry, H2O, H2O(l))")
    return ok


def main():
    checks = [check_C1_nasa9_enthalpy_is_the_integral_of_cp, check_C2_nasa9_internal_energy_joins_the_baseline_at_T0,
              check_C3_h2_rotational_heat_capacity_values, check_C4_h2_internal_energy_is_consistent_with_its_cv,
              check_C5_mayer_relation_in_the_baseline]
    failed = sum(not c() for c in checks)
    print(f"caloric_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
