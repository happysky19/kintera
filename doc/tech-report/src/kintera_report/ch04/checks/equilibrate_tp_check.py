#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Saturation adjustment at fixed temperature and pressure" (chapter 4).

A numpy port, written line for line against equilibrate_tp of src/thermo/equilibrate_tp.h of chengcli/kintera at the
code pin of doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), for one nucleation reaction
V <=> C in a dry carrier D (species order D, V, C; nu = (0, -1, +1)). With one active reaction the KKT solve of
leastsq_kkt_feasible_origin reduces to one division, so the port needs no active-set code; the per-reaction extent
limiter, the halving line search and the renormalisation are ported as they are. The saturation curve is h2o_ideal.
It does not import kintera. Besides its checks it writes ../data/equilibrate_tp_history.json, the iteration history
that fig_equilibrate_tp_convergence draws.

Run: python3 equilibrate_tp_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

DATA = Path(__file__).resolve().parent.parent / "data" / "equilibrate_tp_history.json"
NU = np.array([0.0, -1.0, 1.0])  # D, V, C
NGAS = 2


def logsvp_h2o_ideal(T):
    betal, gammal, betas, gammas, tr, pr = 24.845, 4.986009, 22.98, 0.52, 273.16, 611.7
    t = T / tr
    if T > tr:
        return (1 - 1 / t) * betal - gammal * math.log(t) + math.log(pr)
    return (1 - 1 / t) * betas - gammas * math.log(t) + math.log(pr)


def residual(x, logsvp_shifted):
    """log_frac_sum - (logsvp - A ln p): the kernel's r_j with the shifted L_j."""
    xg = x[:NGAS].sum()
    a = -NU[:NGAS]
    return float((a * np.log(x[:NGAS] / xg))[a != 0].sum() - logsvp_shifted)


def gain(x):
    """sum_i (a_i/x_i - A/x_g) nu_i over the gases: the weight row times the stoichiometric column."""
    xg = x[:NGAS].sum()
    a = -NU[:NGAS]
    A = a.sum()
    w = -A / xg + np.where(a != 0, a / np.where(x[:NGAS] > 0, x[:NGAS], 1.0), 0.0)
    return float((w * NU[:NGAS]).sum())


def equilibrate_tp(x, T, p, eps=1e-6, max_iter=10):
    """The TP kernel for one reaction; returns (x, history, status)."""
    x = x.copy()
    L = logsvp_h2o_ideal(T) - (-NU[:NGAS]).sum() * math.log(p)   # logsvp[j] - stoich_sum ln p
    hist = []
    for it in range(1, max_iter + 1):
        r = residual(x, L)
        prod = x[NGAS:][NU[NGAS:] > 0].prod()                    # condensed products present?
        active = (r < -eps and prod > 0.0) or (r > eps)
        if not active:
            hist.append((it, r, x.copy(), 1.0))
            return x, hist, 0
        rate = -r / gain(x)                                        # the 1x1 KKT solve: gain * rate = rhs = -r
        # per-reaction extent limiter: clip to the stock of what the step consumes
        change = NU * rate
        theta = np.where(-change > x, np.where(x > 0, x / np.where(-change > 0, -change, 1.0), 0.0), 1.0)
        lam = min([theta[i] for i in range(3) if change[i] < 0.0] + [1.0])
        hist.append((it, r, x.copy(), lam))
        rate *= lam
        # halving line search: every gas stays positive (and a positive gas does not hit exactly zero)
        x0 = x.copy()
        scale = 1.0
        while True:
            x = x0 + NU * rate * scale
            if all(x[i] > 0.0 or (x0[i] == 0.0 and x[i] == 0.0) for i in range(NGAS)):
                break
            scale *= 0.5
            if scale < np.finfo(float).tiny:
                return x0, hist, 4
        x /= x.sum()
    return x, hist, 20


def check_C1_gain_is_the_derivative_of_the_residual():
    """[C1] G = dr/dxi along x <- x + nu xi (before renormalisation): symbolic derivative equals the kernel's
    weight-times-stoichiometry form sum_i (a_i/x_i - A/x_g) nu_i."""
    xd, xv, xc, xi = sp.symbols("x_d x_v x_c xi", positive=True)
    xg = xd + (xv - xi)
    r = sp.log((xv - xi) / xg)      # L dropped: constant in xi
    dr = sp.diff(r, xi).subs(xi, 0)
    G = (1 / xv - 1 / (xd + xv)) * (-1)   # (a_v/x_v - A/x_g) nu_v with a_v = 1, A = 1, nu_v = -1
    ok = sp.simplify(dr - G) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} dr/dxi - sum_i (a_i/x_i - A/x_g) nu_i = {sp.simplify(dr - G)}")
    return ok


def check_C2_condensation_converges_quadratically():
    """[C2] From x = (0.9, 0.1, 0) at 280 K and 1 bar (water 10 times supersaturated) the iteration converges to
    |r| < 1e-6 and the ratio |r_{k+1}| / r_k^2 stays bounded (quadratic convergence); the final gas-only vapour
    fraction equals p_sat/p to 1e-10 relative."""
    T, p = 280.0, 1.0e5
    x, hist, status = equilibrate_tp(np.array([0.9, 0.1, 0.0]), T, p, eps=1e-10, max_iter=20)
    rs = [abs(h[1]) for h in hist]
    ratios = [rs[k + 1] / rs[k] ** 2 for k in range(len(rs) - 2) if rs[k] > 1e-8]
    x_sat = math.exp(logsvp_h2o_ideal(T)) / p
    err = abs(x[1] / x[:NGAS].sum() / x_sat - 1)
    ok = status == 0 and err < 1e-10 and max(ratios) < 10.0
    print(f"[C2] {'PASS' if ok else 'FAIL'} condensation: |r| per iteration "
          + ", ".join(f"{v:.2e}" for v in rs) + f"; max |r_k+1|/r_k^2 = {max(ratios):.2f}; "
          f"x_v/x_g - p_sat/p relative error {err:.1e}; cloud x_c = {x[2]:.6f}")
    global HIST_COND
    HIST_COND = hist
    return ok


def check_C3_evaporation_is_clipped_to_the_cloud_stock():
    """[C3] From x = (0.95, 0.001, 0.02) at 300 K and 1 bar (cloud present, strongly undersaturated: the whole water
    inventory 0.021 is below the saturation fraction 0.0354) the Newton step on the logarithmic residual is small at
    first, -r/G with G of order 1/x_v, so the vapour grows by a factor of about 1 + |r| per pass; on the last pass
    the step would evaporate more cloud than exists, the limiter scales the extent to the stock, the cloud reaches
    exactly zero and the reaction leaves the active set (undersaturated, no cloud)."""
    T, p = 300.0, 1.0e5
    x0 = np.array([0.95, 0.001, 0.02])
    L = logsvp_h2o_ideal(T) - math.log(p)
    r0 = residual(x0, L)
    rate_full = -r0 / gain(x0)
    x, hist, status = equilibrate_tp(x0, T, p, eps=1e-10, max_iter=20)
    lams = [h[3] for h in hist]
    growth = [hist[k + 1][2][1] / hist[k][2][1] for k in range(len(hist) - 1)]
    ok = status == 0 and x[2] == 0.0 and min(lams) < 1.0 and residual(x, L) < 0 and abs(rate_full) < x0[2]
    print(f"[C3] {'PASS' if ok else 'FAIL'} evaporation: r_0 = {r0:.3f}, first extent {rate_full:.5f} (cloud stock {x0[2]}); "
          f"vapour growth per pass " + ", ".join(f"{g:.2f}" for g in growth) + f"; limiter factors "
          + ", ".join(f"{l:.3f}" for l in lams) + f"; final x_c = {x[2]}, r = {residual(x, L):.3f}, "
          f"iterations {len(hist)}, status {status}")
    global HIST_EVAP
    HIST_EVAP = hist
    return ok


def check_C4_renormalisation_keeps_mole_ratios():
    """[C4] For a reaction with sum_i nu_i != 0 (MgSiO3: 5 gas moles become 6 species moles) the extent step changes
    sum_i x_i, and the kernel renormalises; the ratio of every species to an inert carrier is unchanged by that
    division, which is what makes element inventories comparable after restoring an amount scale."""
    nu = np.array([0.0, -1.0, -1.0, -3.0, 5.0, 1.0])  # He, Mg, SiH4, H2O, H2, MgSiO3(s)
    x = np.array([0.14, 6e-5, 5e-5, 1e-3, 0.8589, 0.0])
    xi = 2e-5
    x1 = x + nu * xi
    s = x1.sum()
    x2 = x1 / s
    ratios_before = x1[1:] / x1[0]
    ratios_after = x2[1:] / x2[0]
    ok = abs(s - 1) > 1e-6 and np.allclose(ratios_before, ratios_after, rtol=1e-14, atol=0)
    print(f"[C4] {'PASS' if ok else 'FAIL'} sum x after the step = {s:.6f}; ratios to He unchanged by renormalisation "
          f"(max relative change {np.abs(ratios_after / ratios_before - 1).max():.1e})")
    return ok


def check_C5_diag_encoding():
    """[C5] diag = n_iter on success, -(100 status + n_iter) otherwise: status 20 after 10 iterations gives -2010,
    a KKT failure 4 after 3 iterations gives -403; the sign alone separates success from failure."""
    enc = lambda status, n: n if status == 0 else -(100 * status + n)
    ok = enc(0, 3) == 3 and enc(20, 10) == -2010 and enc(4, 3) == -403 and enc(1, 1) == -101
    print(f"[C5] {'PASS' if ok else 'FAIL'} diag: converged in 3 -> {enc(0, 3)}; not converged after 10 -> {enc(20, 10)}; "
          f"KKT error 4 after 3 -> {enc(4, 3)}")
    return ok


def main():
    checks = [check_C1_gain_is_the_derivative_of_the_residual, check_C2_condensation_converges_quadratically,
              check_C3_evaporation_is_clipped_to_the_cloud_stock, check_C4_renormalisation_keeps_mole_ratios,
              check_C5_diag_encoding]
    failed = sum(not c() for c in checks)
    try:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps({
            "provenance": "written by src/kintera_report/ch04/checks/equilibrate_tp_check.py (a numpy port of "
                          "equilibrate_tp at the code pin of doc/tech-report/code_pin.yml); h2o_ideal curve",
            "condensation": {"T_K": 280.0, "p_Pa": 1.0e5, "x0": [0.9, 0.1, 0.0],
                             "iterations": [[h[0], h[1], h[2].tolist(), h[3]] for h in HIST_COND]},
            "evaporation": {"T_K": 300.0, "p_Pa": 1.0e5, "x0": [0.95, 0.001, 0.02],
                            "iterations": [[h[0], h[1], h[2].tolist(), h[3]] for h in HIST_EVAP]},
        }, indent=1) + "\n")
    except NameError:
        failed += 1
    print(f"equilibrate_tp_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
