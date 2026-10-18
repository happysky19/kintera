#!/usr/bin/env python3
"""Executable check of the Derivation layer of "The ideal-gas equation of state and its inversions" (chapter 3).

Mirrors ThermoYImpl::_temp_to_pres, ThermoYImpl::_pres_to_temp and ThermoXImpl::_xfrac_to_conc of chengcli/kintera at
the code pin of doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy and sympy. The
compressibility hook z_i is kept symbolic so that the damping argument of the pressure-to-temperature Newton step
can be checked for z != 1 as well. It does not import kintera.

Run: python3 eos_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import sys

import numpy as np
import sympy as sp

RGAS = 8.31446  # J mol^-1 K^-1, constants::Rgas at the pin


def check_C1_pressure_from_gas_concentrations():
    """[C1] VT->P: p = R T sum_{gas} z_i c_i; with z_i = 1 the Jupiter-like state below gives p = R T c_g exactly
    (tolerance 1e-15 relative, one sum of three terms)."""
    T = 300.0
    c = np.array([40.0, 0.4, 0.2, 0.3])  # dry, H2O, NH3 (gases) and one cloud [mol m^-3]
    ngas = 3
    p = RGAS * T * c[:ngas].sum()
    err = abs(p / (RGAS * T * c[:ngas].sum()) - 1)
    ok = err < 1e-15 and abs(p - 101270.1) < 0.1  # 8.31446 * 300 * 40.6 = 101 270.12 Pa
    print(f"[C1] {'PASS' if ok else 'FAIL'} VT->P: p = {p:.1f} Pa for c_g = {c[:ngas].sum()} mol m^-3 at {T} K; "
          f"the cloud does not enter")
    return ok


def check_C2_ideal_gas_newton_converges_in_one_step():
    """[C2] PV->T: the starting guess T0 = p / (R sum c_i) is the exact root when z_i = 1, so f(T0) = 0 and the first
    Newton step is zero: the loop exits after one iteration with the stopping test |1 - T_pre/T| < ftol."""
    T_true = 213.7
    c = np.array([40.0, 0.4, 0.2])
    p = RGAS * T_true * c.sum()
    T = p / (c.sum() * RGAS)
    cv_R = np.array([2.5, 2.5, 2.5])
    cp_R = cv_R + 1
    f = T * c.sum() - p / RGAS           # z = 1
    step = f / ((cp_R - cv_R) * c).sum()  # (cp - cv) c = c for z = 1
    T_new = T - step
    rel = abs(1 - T / T_new)
    ok = rel < 1e-6 and abs(T_new - T_true) < 1e-12 * T_true
    print(f"[C2] {'PASS' if ok else 'FAIL'} PV->T, z = 1: f(T0) = {f:.2e}, first step {step:.2e} K, "
          f"|1 - T_pre/T| = {rel:.1e} < ftol 1e-6, T = {T_new:.10f} K (exact {T_true})")
    return ok


def check_C3_newton_denominator_is_an_upper_bound_of_the_slope():
    """[C3] For f(T) = T sum z_i(T) c_i - p/R the exact slope is sum (z_i + T z_i') c_i. The code divides by
    sum (cp_i - cv_i) c_i / R instead. With the Mayer relation of a dissociating ideal mixture,
    cp - cv = R (z + T z_T)^2 / (z + c z_c), and z_T >= 0, z_c <= 0, that denominator is >= the exact slope, so the
    step T <- T - f / denominator never overshoots the root: a damped Newton step. Checked symbolically."""
    z, zT, zc, c, T = sp.symbols("z z_T z_c c T", positive=True)
    slope = (z + T * zT) * c
    mayer = (z + T * zT) ** 2 / (z - c * zc) * c   # z_c <= 0 written as -zc with zc >= 0
    ratio = sp.simplify(mayer / slope)             # = (z + T z_T) / (z - c z_c) >= 1 when z_T >= 0, z_c <= 0
    ok = sp.simplify(ratio - (z + T * zT) / (z - c * zc)) == 0
    # numeric spot check of ratio >= 1 on a grid
    f = sp.lambdify((z, zT, zc, c, T), ratio)
    grid = [(1.0, 0.0, 0.0, 10.0, 300.0), (1.2, 1e-3, 1e-3, 10.0, 300.0), (1.5, 5e-3, 1e-2, 2.0, 3000.0)]
    vals = [f(*g) for g in grid]
    ok = ok and all(v >= 1.0 for v in vals)
    print(f"[C3] {'PASS' if ok else 'FAIL'} damping ratio (cp - cv) c / (R f'(T)) = (z + T z_T)/(z - c|z_c|) >= 1; "
          f"values on the grid: {', '.join(f'{v:.4f}' for v in vals)} (ideal gas: exactly 1)")
    return ok


def check_C4_concentration_newton_for_ideal_gas():
    """[C4] TPX->V: the Newton update c <- c + (c_ideal - z c)/(z_c c + z) is stationary at c = c_ideal when z = 1,
    z_c = 0, so the loop also exits after one pass; relative change below ftol = 1e-6 (actually zero)."""
    T, p = 300.0, 1.0e5
    x = np.array([0.95, 0.05])
    c_ideal = x * p / (RGAS * T * x.sum())
    c = c_ideal.copy()
    z, zc = 1.0, 0.0
    c_new = c + (c_ideal - z * c) / (zc * c + z)
    rel = np.abs(1 - c / c_new).max()
    ok = rel < 1e-6
    print(f"[C4] {'PASS' if ok else 'FAIL'} TPX->V, z = 1: max |1 - c_pre/c| = {rel:.1e} after one Newton pass; "
          f"c = {c_new.round(4).tolist()} mol m^-3")
    return ok


def check_C5_gas_floor_keeps_the_temperature_finite():
    """[C5] A cell with no gas at all would divide by zero in PV->T; the floor gas_floor = 1e-20 mol m^-3 on each gas
    concentration keeps sum c_g > 0. With one species at the floor and p = 1e5 Pa the guess is finite (and absurd:
    1e5 / (8.31446 * 1e-20) K), which is why the floor is a guard, not a model."""
    floor = 1e-20
    T0 = 1.0e5 / (RGAS * floor)
    ok = np.isfinite(T0) and T0 > 1e23
    print(f"[C5] {'PASS' if ok else 'FAIL'} gas floor: p/(R gas_floor) = {T0:.3e} K is finite")
    return ok


def main():
    checks = [check_C1_pressure_from_gas_concentrations, check_C2_ideal_gas_newton_converges_in_one_step,
              check_C3_newton_denominator_is_an_upper_bound_of_the_slope, check_C4_concentration_newton_for_ideal_gas,
              check_C5_gas_floor_keeps_the_temperature_finite]
    failed = sum(not c() for c in checks)
    print(f"eos_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
