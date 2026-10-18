#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Saturation adjustment at fixed energy and volume" (chapter 4).

A numpy port, written against equilibrate_uv of src/thermo/equilibrate_uv.h of chengcli/kintera at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), for one nucleation reaction V <=> C in a
dry carrier D with the constant heat capacities and offsets of tests/jupiter.yaml (cv_R 2.5, 2.5, 9.0; u*/R
-750 K, -750 K, -6130 K after the shift of chapter 3) and the h2o_ideal curve. Concentrations in mol m^-3,
energies in J m^-3. It does not import kintera. It writes ../data/equilibrate_uv_history.json for
fig_equilibrate_uv_convergence.

Run: python3 equilibrate_uv_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

DATA = Path(__file__).resolve().parent.parent / "data" / "equilibrate_uv_history.json"
RGAS = 8.31446
NU = np.array([0.0, -1.0, 1.0])
NGAS = 2
CV = np.array([2.5, 2.5, 9.0]) * RGAS           # J mol^-1 K^-1  (cv_const)
U0 = np.array([-750.0, -750.0, -6130.0]) * RGAS  # J mol^-1       (intEng_offset)


def logsvp(T):
    betal, gammal, betas, gammas, tr, pr = 24.845, 4.986009, 22.98, 0.52, 273.16, 611.7
    t = T / tr
    if T > tr:
        return (1 - 1 / t) * betal - gammal * math.log(t) + math.log(pr)
    return (1 - 1 / t) * betas - gammas * math.log(t) + math.log(pr)


def logsvp_ddT(T):
    betal, gammal, betas, gammas, tr = 24.845, 4.986009, 22.98, 0.52, 273.16
    t = T / tr
    if T > tr:
        return (betal / (t * t) - gammal / t) / tr
    return (betas / (t * t) - gammas / t) / tr


def energy(c, T):
    return float(((U0 + CV * T) * c).sum())


def equilibrate_uv(c, T, eps=1e-6, max_iter=10):
    """The UV kernel (KKT branch) for one reaction; returns (c, T, history, status)."""
    c = c.copy()
    h0 = energy(c, T)
    A = (-NU[:NGAS]).sum()
    hist = []
    for it in range(1, max_iter + 1):
        L = logsvp(T) - A * math.log(RGAS * T)          # logsvp[j] in concentration units
        Lp = logsvp_ddT(T) - A / T                       # logsvp_ddT[j]
        u = U0 + CV * T                                  # intEng[i]
        heat_capacity = float((CV * c).sum())
        a = -NU[:NGAS]
        log_conc_sum = float((a * np.log(c[:NGAS]))[a != 0].sum())
        r = log_conc_sum - L
        prod = c[NGAS:][NU[NGAS:] > 0].prod()
        active = (r < -eps and prod > 0.0) or (r > eps)
        hist.append((it, T, r, energy(c, T) - h0, c.copy()))
        if not active:
            return c, T, hist, 0
        # weight row: logsvp_ddT * u_i / C_V + a_i / c_i (gases); times nu gives the gain
        w = Lp * u / heat_capacity
        w[:NGAS] += np.where(a != 0, a / c[:NGAS], 0.0)
        G = float((w * NU).sum())
        rate = -r / G
        # extent limiter and the 0.99 backtracking of the gas ratio
        change = NU * rate
        theta = np.where(-change > c, np.where(c > 0, c / np.where(-change > 0, -change, 1.0), 0.0), 1.0)
        lam = min([theta[i] for i in range(3) if change[i] < 0.0] + [1.0])
        rate *= lam
        c0 = c.copy()
        scale = 1.0
        while True:
            c = c0 + NU * rate * scale
            good = all(not (c0[i] > 0 and (c[i] / c0[i] > 100.0 or c[i] / c0[i] < 0.01)) for i in range(NGAS))
            if good:
                break
            scale *= 0.99
        # scalar caloric Newton for T at the new composition
        T0 = 0.0
        n = 0
        while abs(T - T0) > max(1e-4, 8 * np.finfo(float).eps * abs(T)):
            n += 1
            if n > 100:
                return c, T, hist, 4
            zh = energy(c, T)
            zc = float((CV * c).sum())
            T0 = T
            T += (h0 - zh) / zc
    return c, T, hist, 20


def check_C1_weight_from_eliminating_temperature():
    """[C1] The coupled Newton system [[D nu, b], [u^T nu, C_V]] [dxi, dT] = -[r, U - U0] with b_j = A_j/T - L'_j
    gives, when the energy equation already holds, (D nu - b u^T nu / C_V) dxi = -r, i.e. the reduced weight
    W = D + (L' - A/T) u^T / C_V: the kernel's logsvp_ddT * u_i / C_V + a_i / c_i. Symbolic, one reaction, two species."""
    D1, D2, u1, u2, CV_, Lp, A, T, n1, n2 = sp.symbols("D_1 D_2 u_1 u_2 C_V L_p A T nu_1 nu_2")
    b = A / T - Lp
    Dnu = D1 * n1 + D2 * n2
    unu = u1 * n1 + u2 * n2
    reduced = Dnu - b * unu / CV_
    W = sp.Matrix([[D1 + (Lp - A / T) * u1 / CV_, D2 + (Lp - A / T) * u2 / CV_]])
    Wnu = (W * sp.Matrix([n1, n2]))[0, 0]
    ok = sp.simplify(reduced - Wnu) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} reduced Jacobian minus W^UV nu: {sp.simplify(reduced - Wnu)}")
    return ok


def check_C2_condensation_at_fixed_energy():
    """[C2] From c = (36, 4, 0) mol m^-3 at 280 K (water 10 times supersaturated at that density) the KKT branch
    converges to |r| < 1e-6 in a few iterations; the final state is saturated at the final temperature, the energy
    is conserved to 1e-12 relative, and the temperature rises (latent heat released)."""
    c0, T0 = np.array([36.0, 4.0, 0.0]), 280.0
    c, T, hist, status = equilibrate_uv(c0, T0, eps=1e-10, max_iter=20)
    h0 = energy(c0, T0)
    e_rel = abs(energy(c, T) - h0) / abs(h0)
    L = logsvp(T) - math.log(RGAS * T)
    r_final = math.log(c[1]) - L
    ok = status == 0 and e_rel < 1e-12 and abs(r_final) < 1e-9 and T > T0
    print(f"[C2] {'PASS' if ok else 'FAIL'} UV condensation: iterations {len(hist)}, T {T0} -> {T:.4f} K, "
          f"|r| history " + ", ".join(f"{abs(h[2]):.1e}" for h in hist) + f"; energy error {e_rel:.1e}; "
          f"cloud {c[2]:.4f} mol m^-3")
    global HIST
    HIST = (c0, T0, hist, status)
    return ok


def check_C3_temperature_loop_tolerance():
    """[C3] The caloric inversion stops when |T - T_prev| <= max(1e-4 K, 8 eps |T|): at 300 K that is 1e-4 K in double
    (8 eps T = 5e-13) and 2.9e-4 K in single precision (8 eps T with eps = 1.19e-7), so float32 cannot oscillate
    below its own resolution."""
    T = 300.0
    tol64 = max(1e-4, 8 * np.finfo(np.float64).eps * T)
    tol32 = max(1e-4, 8 * np.finfo(np.float32).eps * T)
    ok = abs(tol64 - 1e-4) < 1e-18 and abs(tol32 - 8 * np.finfo(np.float32).eps * T) < 1e-12 and tol32 > tol64
    print(f"[C3] {'PASS' if ok else 'FAIL'} temperature tolerance at 300 K: double {tol64:.2e} K, single {tol32:.2e} K")
    return ok


def check_C4_backtracking_factor():
    """[C4] The gas ratio after a step must stay within [0.01, 100] of its start; the step is scaled by 0.99 until it
    does, a linear (not geometric) shortening. A step that would take a vapour to one thousandth, or exactly to zero,
    is accepted after one reduction (ratio 0.011 or 0.01); one that would overshoot to -1 times the start needs
    ceil(ln(0.495)/ln(0.99)) = 70; one that would multiply by 1000 needs ceil(ln(0.0991)/ln(0.99)) = 230. The
    extent limiter, applied first, removes the overshoots the first two cases stand for."""
    def reductions(target):
        c0, scale, n = 1.0, 1.0, 0
        while True:
            c = c0 + (target - c0) * scale
            if 0.01 <= c / c0 <= 100.0:
                return n, scale
            scale *= 0.99
            n += 1
    got = {t: reductions(t)[0] for t in (1e-3, 0.0, -1.0, 1000.0)}
    want = {1e-3: 1, 0.0: 1, -1.0: math.ceil(math.log(0.495) / math.log(0.99)),
            1000.0: math.ceil(math.log(99.0 / 999.0) / math.log(0.99))}
    ok = got == want
    print(f"[C4] {'PASS' if ok else 'FAIL'} 0.99 backtracking, reductions for a step to 1e-3, 0, -1, 1000 times the start: "
          + ", ".join(str(got[t]) for t in (1e-3, 0.0, -1.0, 1000.0)))
    return ok


def check_C5_sign_of_the_temperature_change():
    """[C5] Condensing dc mol m^-3 of vapour at fixed U and V changes the temperature by
    dT = -(u_c - u_v) dc / C_V = l dc / C_V > 0 (l = u_v - u_c > 0): for 1 mol m^-3 of water in the state of [C2],
    about l / C_V with l = 44.7 kJ mol^-1 and C_V = 831 J m^-3 K^-1 at 280 K."""
    T = 280.0
    c = np.array([36.0, 4.0, 0.0])
    l = (U0[1] + CV[1] * T) - (U0[2] + CV[2] * T)
    CVtot = float((CV * c).sum())
    dT = l * 1.0 / CVtot
    ok = l > 0 and dT > 0
    print(f"[C5] {'PASS' if ok else 'FAIL'} l = u_v - u_c = {l:.0f} J mol^-1, C_V = {CVtot:.0f} J m^-3 K^-1, "
          f"dT per mol m^-3 condensed = {dT:.2f} K")
    return ok


def main():
    checks = [check_C1_weight_from_eliminating_temperature, check_C2_condensation_at_fixed_energy,
              check_C3_temperature_loop_tolerance, check_C4_backtracking_factor,
              check_C5_sign_of_the_temperature_change]
    failed = sum(not c() for c in checks)
    try:
        c0, T0, hist, status = HIST
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps({
            "provenance": "written by src/kintera_report/ch04/checks/equilibrate_uv_check.py (a numpy port of "
                          "equilibrate_uv, KKT branch, at the code pin of doc/tech-report/code_pin.yml); "
                          "h2o_ideal curve, tests/jupiter.yaml heat capacities",
            "c0_mol_m3": c0.tolist(), "T0_K": T0, "status": status,
            "iterations": [[h[0], h[1], h[2], h[3], h[4].tolist()] for h in hist],
        }, indent=1) + "\n")
    except NameError:
        failed += 1
    print(f"equilibrate_uv_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
