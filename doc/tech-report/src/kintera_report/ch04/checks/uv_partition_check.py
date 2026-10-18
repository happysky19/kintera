#!/usr/bin/env python3
"""Executable check of the Derivation layer of "The partition solver for disjoint vapour-cloud pairs" (chapter 4).

A numpy port of partition_uv_state and equilibrate_uv_partition of src/thermo/equilibrate_uv.h of chengcli/kintera
at the code pin of doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), for one water pair in a
dry carrier with the heat capacities of tests/jupiter.yaml and the h2o_ideal curve; it reuses the state of
equilibrate_uv_check.py so that the two solvers can be compared. It does not import kintera. It writes
../data/uv_partition_energy.json for fig_uv_partition_energy.

Run: python3 uv_partition_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equilibrate_uv_check import CV, NGAS, NU, RGAS, U0, energy, equilibrate_uv, logsvp, logsvp_ddT  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "uv_partition_energy.json"


def partition_state(T, baseline, total):
    """partition_uv_state for one pair with a = b = 1: the vapour is min(total, c_sat(T)), the cloud the rest;
    returns (c, energy, derivative)."""
    c = baseline.copy()
    log_sat = logsvp(T) / 1.0 - math.log(RGAS * T)
    vapour = total if log_sat >= math.log(total) else math.exp(log_sat)
    vapour = min(vapour, total)
    c[1], c[2] = vapour, (total - vapour) * 1.0
    e = energy(c, T)
    d = float((CV * c).sum())
    if c[1] < total:
        d += c[1] * (logsvp_ddT(T) / 1.0 - 1.0 / T) * ((U0[1] + CV[1] * T) - 1.0 * (U0[2] + CV[2] * T))
    return c, e, d


def equilibrate_partition(c0, T0, max_iter=10):
    """equilibrate_uv_partition for one pair; returns (c, T, iterations, bracket, status, trace)."""
    h0 = energy(c0, T0)
    total = c0[1] + c0[2]
    tol = 64 * np.finfo(float).eps * max(abs(h0), 1.0)
    c, e, d = partition_state(T0, c0, total)
    res = e - h0
    iterations = 1
    trace = [(T0, res)]
    lower = upper = T0
    if abs(res) > tol:
        bracketed = False
        for _ in range(64):
            cand = lower * 0.5 if res > 0 else upper * 2.0
            cb, eb, db = partition_state(cand, c0, total)
            if res > 0:
                lower, bracketed = cand, eb <= h0
            else:
                upper, bracketed = cand, eb >= h0
            if bracketed:
                break
        if not bracketed:
            return c0, T0, iterations, (lower, upper), -1, trace
        cur = T0
        while abs(res) > tol and iterations < max_iter:
            cand = cur - res / d
            if not math.isfinite(cand) or cand <= lower or cand >= upper:
                cand = lower + 0.5 * (upper - lower)
            if cand == cur:
                cand = lower + 0.5 * (upper - lower)
            if cand == cur:
                break
            c, e, d = partition_state(cand, c0, total)
            cur = cand
            res = e - h0
            trace.append((cur, res))
            if res < 0:
                lower = cur
            else:
                upper = cur
            iterations += 1
        T = cur
    else:
        T = T0
    status = 0 if abs(res) <= tol else -1
    return c, T, iterations, (lower, upper), status, trace


def check_C1_saturation_concentration_derivative():
    """[C1] For c_sat(T) = exp(S(T)/a) / (R T), dc_sat/dT = c_sat (S'(T)/a - 1/T): the slope term the partition solver
    adds for every pair that is below its inventory. Symbolic."""
    T, a = sp.symbols("T a", positive=True)
    S = sp.Function("S")
    R = sp.Symbol("R", positive=True)
    c = sp.exp(S(T) / a) / (R * T)
    diff = sp.simplify(sp.diff(c, T) - c * (sp.diff(S(T), T) / a - 1 / T))
    ok = diff == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} dc_sat/dT - c_sat (S'/a - 1/T) = {diff}")
    return ok


def check_C2_partition_agrees_with_the_kkt_branch():
    """[C2] From c = (36, 4, 0) mol m^-3 at 280 K the partition solver and the KKT port of equilibrate_uv_check reach
    the same temperature and composition: agreement to 1e-7 relative in T and in the cloud amount (the KKT branch
    stops on its 1e-10 log-quotient tolerance, the partition branch on 64 eps in energy)."""
    c0, T0 = np.array([36.0, 4.0, 0.0]), 280.0
    cp, Tp, it, bracket, status, trace = equilibrate_partition(c0, T0, max_iter=20)
    ck, Tk, hist, status_k = equilibrate_uv(c0, T0, eps=1e-10, max_iter=20)
    ok = status == 0 and status_k == 0 and abs(Tp / Tk - 1) < 1e-7 and abs(cp[2] / ck[2] - 1) < 1e-7
    print(f"[C2] {'PASS' if ok else 'FAIL'} partition T = {Tp:.6f} K in {it} iterations (bracket {bracket[0]:.2f}-{bracket[1]:.2f} K), "
          f"KKT T = {Tk:.6f} K; cloud {cp[2]:.6f} vs {ck[2]:.6f} mol m^-3")
    global TRACE
    TRACE = (c0, T0, trace, cp, Tp)
    return ok


def check_C3_inventory_and_energy_at_convergence():
    """[C3] The pair inventory N = c_V + (a/b) c_C is unchanged by the partition and the energy residual at
    convergence is within 64 eps max(|U_0|, 1)."""
    c0, T0 = np.array([36.0, 4.0, 0.0]), 280.0
    c, T, it, bracket, status, trace = equilibrate_partition(c0, T0, max_iter=20)
    h0 = energy(c0, T0)
    tol = 64 * np.finfo(float).eps * max(abs(h0), 1.0)
    ok = abs((c[1] + c[2]) - (c0[1] + c0[2])) < 1e-13 and abs(energy(c, T) - h0) <= tol
    print(f"[C3] {'PASS' if ok else 'FAIL'} inventory {c0[1] + c0[2]} -> {c[1] + c[2]:.15f} mol m^-3; "
          f"|U - U_0| = {abs(energy(c, T) - h0):.2e} J m^-3 (tolerance {tol:.2e})")
    return ok


def check_C4_eligibility_rule():
    """[C4] A network is partitionable iff every reaction has exactly one gas reactant and one condensed product and
    no species appears in two reactions (ThermoYImpl::reset). jupiter.yaml (NH3 in two reactions): no; the CH4/H2S
    pairs: yes; Mn + H2S -> MnS(s) + H2 (two gas reactants, a gas product): no."""
    def partitionable(stoich, ngas):
        used = set()
        for j in range(stoich.shape[1]):
            v = c = 0
            for i in range(stoich.shape[0]):
                k = stoich[i, j]
                if k == 0:
                    continue
                if i in used:
                    return False
                used.add(i)
                if i < ngas and k < 0:
                    v += 1
                elif i >= ngas and k > 0:
                    c += 1
                else:
                    return False
            if v != 1 or c != 1:
                return False
        return stoich.shape[1] > 0

    jup = np.zeros((7, 3))  # dry H2O NH3 H2S | H2O(l) NH3(s) NH4SH(s)
    jup[1, 0], jup[4, 0] = -1, 1
    jup[2, 1], jup[5, 1] = -1, 1
    jup[2, 2], jup[3, 2], jup[6, 2] = -1, -1, 1
    pairs = np.zeros((5, 2))  # dry CH4 H2S | CH4(s) H2S(s)
    pairs[1, 0], pairs[3, 0] = -1, 1
    pairs[2, 1], pairs[4, 1] = -1, 1
    mns = np.zeros((5, 1))  # He Mn H2S H2 | MnS(s)
    mns[1, 0], mns[2, 0], mns[3, 0], mns[4, 0] = -1, -1, 1, 1
    got = (partitionable(jup, 4), partitionable(pairs, 3), partitionable(mns, 4))
    ok = got == (False, True, False)
    print(f"[C4] {'PASS' if ok else 'FAIL'} partitionable: jupiter.yaml {got[0]}, CH4/H2S pairs {got[1]}, MnS with H2 {got[2]}")
    return ok


def main():
    checks = [check_C1_saturation_concentration_derivative, check_C2_partition_agrees_with_the_kkt_branch,
              check_C3_inventory_and_energy_at_convergence, check_C4_eligibility_rule]
    failed = sum(not c() for c in checks)
    try:
        c0, T0, trace, cp, Tp = TRACE
        h0 = energy(c0, T0)
        total = c0[1] + c0[2]
        Ts = np.linspace(260.0, 320.0, 241)
        F = [partition_state(float(t), c0, total)[1] - h0 for t in Ts]
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps({
            "provenance": "written by src/kintera_report/ch04/checks/uv_partition_check.py (a numpy port of "
                          "equilibrate_uv_partition at the code pin of doc/tech-report/code_pin.yml); h2o_ideal "
                          "curve, tests/jupiter.yaml heat capacities",
            "c0_mol_m3": c0.tolist(), "T0_K": T0, "T_solution_K": Tp,
            "curve": {"T_K": Ts.tolist(), "F_J_m3": F},
            "iterates": [[t, r] for t, r in trace],
        }, indent=1) + "\n")
    except NameError:
        failed += 1
    print(f"uv_partition_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
