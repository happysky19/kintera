#!/usr/bin/env python3
"""Executable check of the Derivation layer of "Species, phases and the two state vectors" (chapter 3).

Mirrors, formula by formula, the conversions of ThermoYImpl::_yfrac_to_xfrac, ThermoYImpl::_yfrac_to_ivol,
ThermoYImpl::_ivol_to_yfrac, ThermoXImpl::_xfrac_to_yfrac and ThermoXImpl::_xfrac_to_conc of chengcli/kintera at
the code pin of doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), written in numpy and sympy.
It does not import kintera.

Run: python3 species_state_check.py   (needs numpy and sympy; exits with the number of failed checks)
"""
import sys

import numpy as np
import sympy as sp

RGAS = 8.31446  # J mol^-1 K^-1, constants::Rgas at the pin


def check_C1_mass_to_mole_fraction_closed_form():
    """[C1] The mole fraction of species i >= 1 from the mass fractions, as _yfrac_to_xfrac evaluates it,
    equals the textbook x_i = (y_i/M_i) / sum_k (y_k/M_k) with y_0 = 1 - sum_{k>=1} y_k, symbolically (3 species)."""
    y1, y2, M0, M1, M2 = sp.symbols("y1 y2 M0 M1 M2", positive=True)
    y0 = 1 - y1 - y2
    # textbook
    denom = y0 / M0 + y1 / M1 + y2 / M2
    x1_text = (y1 / M1) / denom
    # the code: x_i = y_i * (M0/M_i) / (1 + sum_k y_k (M0/M_k - 1)), i >= 1
    s = 1 + y1 * (M0 / M1 - 1) + y2 * (M0 / M2 - 1)
    x1_code = y1 * (M0 / M1) / s
    x0_code = 1 - x1_code - y2 * (M0 / M2) / s
    ok = sp.simplify(x1_text - x1_code) == 0 and sp.simplify((y0 / M0) / denom - x0_code) == 0
    print(f"[C1] {'PASS' if ok else 'FAIL'} _yfrac_to_xfrac closed form equals (y_i/M_i)/sum(y_k/M_k): "
          f"difference simplifies to {sp.simplify(x1_text - x1_code)}")
    return ok


def check_C2_mole_to_mass_fraction_is_the_inverse():
    """[C2] _xfrac_to_yfrac (y_i = x_i M_i / sum_k x_k M_k) inverts _yfrac_to_xfrac to round-off on a random
    5-species state; tolerance 1e-14 relative: a few operations on O(1) numbers in double precision."""
    rng = np.random.default_rng(3)
    M = np.array([2.3e-3, 18e-3, 17e-3, 18e-3, 17e-3])
    y = rng.uniform(0, 0.05, 4)
    y0 = 1 - y.sum()
    s = 1 + (y * (M[0] / M[1:] - 1)).sum()
    x = np.empty(5)
    x[1:] = y * (M[0] / M[1:]) / s
    x[0] = 1 - x[1:].sum()
    y_back = (x * M)[1:] / (x * M).sum()
    err = np.abs(y_back - y).max() / np.abs(y).max()
    ok = err < 1e-14 and abs(y0 - (x * M)[0] / (x * M).sum()) < 1e-14
    print(f"[C2] {'PASS' if ok else 'FAIL'} X->Y inverts Y->X: max relative error {err:.2e} (tolerance 1e-14)")
    return ok


def check_C3_partial_density_round_trip():
    """[C3] DY->V (rho_i = rho y_i, rho_0 = rho (1 - sum y)) followed by V->Y (y_i = rho_i / sum rho_k) returns the
    mass fractions, and the partial densities sum to rho; tolerance 1e-15 relative (one sum of 5 terms)."""
    rng = np.random.default_rng(5)
    rho = 0.37
    y = rng.uniform(0, 0.05, 4)
    v = np.concatenate([[rho * (1 - y.sum())], rho * y])
    ok1 = abs(v.sum() - rho) / rho < 1e-15
    y_back = v[1:] / v.sum()
    ok2 = np.abs(y_back - y).max() < 1e-15
    print(f"[C3] {'PASS' if ok1 and ok2 else 'FAIL'} DY->V then V->Y round trip: sum(rho_i) - rho = {v.sum() - rho:.2e}, "
          f"max |y' - y| = {np.abs(y_back - y).max():.2e} (tolerance 1e-15)")
    return ok1 and ok2


def check_C4_ideal_gas_concentration_uses_the_gas_total():
    """[C4] TPX->V for an ideal gas: c_i = x_i p / (R T x_g) for gases, so that sum_{gas} c_i = p/(R T) whatever the
    condensate load; condensates get c_i = c_0 x_i / x_0. Tolerance 1e-14 relative."""
    T, p = 250.0, 1.0e5
    x = np.array([0.90, 0.03, 0.02, 0.04, 0.01])  # dry, two vapours, two clouds
    ngas = 3
    xg = x[:ngas].sum()
    c = np.empty_like(x)
    c[:ngas] = x[:ngas] * p / (RGAS * T * xg)
    c[ngas:] = c[0] * x[ngas:] / x[0]
    err1 = abs(c[:ngas].sum() * RGAS * T / p - 1)
    err2 = np.abs(c[ngas:] / c[0] - x[ngas:] / x[0]).max()
    ok = err1 < 1e-14 and err2 < 1e-14
    print(f"[C4] {'PASS' if ok else 'FAIL'} sum_gas c_i R T / p - 1 = {err1:.2e}; cloud ratio error {err2:.2e} "
          f"(tolerance 1e-14); gas total {c[:ngas].sum():.6f} mol m^-3 at {T} K, {p:.0f} Pa")
    return ok


def main():
    checks = [check_C1_mass_to_mole_fraction_closed_form, check_C2_mole_to_mass_fraction_is_the_inverse,
              check_C3_partial_density_round_trip, check_C4_ideal_gas_concentration_uses_the_gas_total]
    failed = sum(not c() for c in checks)
    print(f"species_state_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
