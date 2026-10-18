#!/usr/bin/env python3
"""Executable check of the Derivation layer of "The batched drivers: warm starts, diagnostics and failure reporting"
(chapter 4).

Mirrors the diagnostic encoding, the warm-start validity test, the gain scatter and the per-cell workspace budget of
ThermoXImpl::forward, ThermoYImpl::forward, equilibrate_tp and equilibrate_uv of chengcli/kintera at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45), in numpy. It does not import kintera.

Run: python3 drivers_check.py   (needs numpy; exits with the number of failed checks)
"""
import sys

import numpy as np


def check_C1_diag_decoding_and_failure_count():
    """[C1] diag >= 0 is the iteration count of a converged cell; diag < 0 encodes -(100 status + iterations) with
    status 1 (invalid input), 2 (forced partition failed), 4 (KKT or temperature failure) or 20 (iteration cap);
    the driver counts the failures as the number of cells with diag < 0. Decoding a sample array."""
    diag = np.array([3.0, 1.0, -2010.0, -403.0, -100.0, 7.0])
    nfail = int((diag < 0).sum())
    decoded = [(int(-d) // 100, int(-d) % 100) if d < 0 else (0, int(d)) for d in diag]
    ok = nfail == 3 and decoded[2] == (20, 10) and decoded[3] == (4, 3) and decoded[4] == (1, 0)
    print(f"[C1] {'PASS' if ok else 'FAIL'} diag {diag.tolist()} -> (status, iterations) {decoded}; failures {nfail}")
    return ok


def check_C2_warm_start_validity():
    """[C2] A warm start reuses the saved active set only when it exists, lives on the same device and has the shape
    (cells, reactions) of the call; otherwise the driver cold-starts with the identity order and nactive = 0."""
    def reuse(warm, saved_shape, saved_device, shape, device, nreaction):
        return warm and saved_shape is not None and saved_device == device and saved_shape == shape + (nreaction,)

    cases = [(True, (2, 3, 7), "cpu", (2, 3), "cpu", 7, True), (True, (2, 3, 7), "cpu", (8, 8, 8), "cpu", 7, False),
             (True, (2, 3, 7), "cpu", (2, 3), "cuda", 7, False), (False, (2, 3, 7), "cpu", (2, 3), "cpu", 7, False),
             (True, None, "cpu", (2, 3), "cpu", 7, False)]
    ok = all(reuse(*c[:6]) == c[6] for c in cases)
    print(f"[C2] {'PASS' if ok else 'FAIL'} warm start reused in {sum(reuse(*c[:6]) for c in cases)} of {len(cases)} cases "
          "(same shape and device only)")
    return ok


def check_C3_gain_scatter_onto_the_active_block():
    """[C3] The kernels compute the gain as an (nactive x nactive) block in active order and scatter it to the
    reaction order through reaction_set; rows and columns of inactive reactions stay zero (the contract of
    tests/test_equilibrate_uv_gain.py). nreaction = 3, active = [2, 0]."""
    nreaction, reaction_set, nactive = 3, [2, 0, 1], 2
    block = np.array([[1.5, -0.2], [0.3, 2.5]])  # (nactive x nactive), row-major, leading dimension nactive
    gain = np.zeros((nreaction, nreaction))
    for k in range(nactive):
        for l in range(nactive):
            gain[reaction_set[k], reaction_set[l]] = block[k, l]
    ok = gain[1].any() == False and gain[:, 1].any() == False and gain[2, 2] == 1.5 and gain[0, 0] == 2.5 and gain[2, 0] == -0.2
    print(f"[C3] {'PASS' if ok else 'FAIL'} scattered gain:\n{gain}")
    return ok


def _align8(nbytes):
    """pool_allocation_bytes of src/utils/alloc.h: every allocation rounded up to 8 bytes, an empty one to 8."""
    nbytes = 8 if nbytes == 0 else nbytes
    return (nbytes + 7) // 8 * 8


def _kkt_feasible_origin_space(n2, n3, t=8, i=4):
    """leastsq_kkt_feasible_origin_space<T>(n2, n3) of src/utils/alloc.h, line for line: the direct-solve hook
    (at_lu, at_col, at_indx), leastsq_kkt_space (aug, ata, column_norm, rhs, eval, ct_indx, lu_indx, skip_row) and
    ludcmp_space(n2 + n3) (vv); T of t bytes, int of i bytes."""
    size = n2 + n3
    hook = _align8(n2 * n2 * t) + _align8(n2 * t) + _align8(n2 * i)
    kkt = (_align8(size * size * t) + _align8(n2 * n2 * t) + _align8(n2 * t) + _align8(size * t) + _align8(n3 * t)
           + _align8(n3 * i) + _align8(size * i) + _align8(size * i))
    return hook + kkt + _align8(size * t)


def check_C4_workspace_budget_of_one_cell():
    """[C4] The per-cell workspace of equilibrate_tp_space<double>(nspecies, nreaction) of src/utils/alloc.h is
    nreaction (logsvp) + nreaction*nspecies (weight) + nreaction (rhs) + nspecies*nreaction (stoich_active) +
    nreaction (stoich_sum) + nspecies (xfrac0) + nreaction^2 (gain_cpy) + nspecies (theta) doubles, i.e.
    8 (3 nr + 2 nr ns + 2 ns + nr^2) bytes, plus the constrained least-squares buffers with nr unknowns and ns
    bounds; equilibrate_uv_space adds intEng, intEng_ddT and logsvp_ddT and drops stoich_sum. For the eight
    vapour-cloud pairs of tests/test_cuda_workspace.py (nspecies 17, nreaction 8) the TP budget is 10.2 KB per cell
    and 325 KB for a 32-thread block, the "about 10 KB" and "325 KB" of that test's description."""
    ns, nr = 17, 8
    own_tp = nr + nr * ns + nr + ns * nr + nr + ns + nr * nr + ns
    kkt = _kkt_feasible_origin_space(nr, ns)
    w_tp = 8 * own_tp + kkt
    own_uv = ns + ns + nr + nr + nr * ns + nr + ns * nr + ns + nr * nr + ns
    w_uv = 8 * own_uv + kkt
    ok = (own_tp == 3 * nr + 2 * nr * ns + 2 * ns + nr * nr == 394 and 8 * own_tp == 3152 and kkt == 7000
          and w_tp == 10152 and w_uv == 10424 and round(32 * w_tp / 1000) == 325)
    print(f"[C4] {'PASS' if ok else 'FAIL'} workspace per cell for nspecies {ns}, nreaction {nr}, double: "
          f"equilibrate_tp {8 * own_tp} B own + {kkt} B KKT = {w_tp} B ({w_tp / 1000:.1f} KB), "
          f"equilibrate_uv {w_uv} B ({w_uv / 1000:.1f} KB); 32-thread block {32 * w_tp / 1000:.0f} KB (TP)")
    return ok


def main():
    checks = [check_C1_diag_decoding_and_failure_count, check_C2_warm_start_validity,
              check_C3_gain_scatter_onto_the_active_block, check_C4_workspace_budget_of_one_cell]
    failed = sum(not c() for c in checks)
    print(f"drivers_check: {len(checks) - failed} of {len(checks)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
