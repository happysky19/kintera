"""Paired CPU/CUDA `ThermoX.forward` at two `ftol` settings, on one state.

    python -P src/kintera_report/ch12/runs/ftol_sweep.py --ftol 1e-12
    python -P src/kintera_report/ch12/runs/ftol_sweep.py --ftol 1e-5

This is the driver behind `^ch12-dispatch-ftol32`. The footnote is unusually complete, and this
follows it exactly: the synthetic eight-vapour card
`derivation-checks/ch12/saved-preparation/one-cell.yaml` (17 species, 8 reactions, hence 17 mole
fractions and 64 gain entries per cell), 200 cells over 250 to 400 K, the isolated cell is cell 1 at
T = 250.75376892089844 K and 1e5 Pa, `float32`, `max-iter` 50. It runs `ThermoX.forward` on CPU and
on CUDA tensors of the same inputs and compares the diagnostics, the 17 mole fractions and the 64
gain entries at atol 1e-6, rtol 1e-4.

What the chapter reports, and what this prints so it can be checked:

  ftol 1e-12   diagnostics 11 (CPU) and -2050 (CUDA); fractions differ by at most 3.73e-9 with
               0 entries outside tolerance; gain differs by up to 5110 with 15 of 64 outside.
  ftol 1e-5    both diagnostics 11; fractions differ by at most 1.86e-9.

The footnote calls this "one pair of builds, one state, not a sweep over states or devices; a
measurement outside the suite, not a committed check", and this script does not change that: it
makes the measurement repeatable, it does not turn it into a contract. It exits non-zero only when
it cannot make the measurement at all.
"""
import argparse
import os
import sys

CELLS = 200
T_LO, T_HI = 250.0, 400.0
CELL = 1
T_CELL = 250.75376892089844
P_CELL = 1e5
MAX_ITER = 50
ATOL, RTOL = 1e-6, 1e-4
DECK = "one-cell.yaml"


def _deck() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    saved = os.environ.get("CH12_SAVED") or os.path.join(
        here, "..", "..", "..", "..", "derivation-checks", "ch12", "saved-preparation")
    path = os.path.normpath(os.path.join(saved, DECK))
    if not os.path.exists(path):
        sys.exit(f"ftol_sweep.py: no deck at {path} (set CH12_SAVED)")
    return path


def run(torch, kintera, device, ftol, dtype):
    op = kintera.ThermoOptions.from_yaml(_deck())
    op.max_iter(MAX_ITER)
    op.ftol(ftol)
    th = kintera.ThermoX(op).to(device)
    nsp = len(op.species())
    T = torch.linspace(T_LO, T_HI, CELLS, dtype=dtype)
    T[CELL] = T_CELL
    P = torch.full((CELLS,), P_CELL, dtype=dtype)
    X = torch.zeros((CELLS, nsp), dtype=dtype)
    for vid in op.vapor_ids():
        X[:, vid] = 1e-3
    X[:, 0] = 1.0 - X.sum(dim=1)
    out = th.forward(T.to(device), P.to(device), X.to(device))
    gain = th.buffer("gain").cpu() if "gain" in dir(th) else None
    diag = th.buffer("diagnostics").cpu() if "diagnostics" in dir(th) else None
    return out.cpu(), gain, diag


def compare(torch, a, b, label):
    if a is None or b is None:
        return f"{label}: not exposed by this build"
    d = (a.double() - b.double()).abs()
    tol = ATOL + RTOL * b.double().abs()
    outside = int((d > tol).sum())
    return f"{label}: max abs difference {float(d.max()):.6g}, {outside} of {d.numel()} outside tolerance"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ftol", type=float, default=1e-12)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    try:
        import torch
        import kintera
    except ImportError as e:
        sys.exit(f"ftol_sweep.py: {e}. Build and install kintera first.")
    if not torch.cuda.is_available():
        print("ftol_sweep.py: SKIPPED, no CUDA device visible; this measurement is a CPU/CUDA pair",
              file=sys.stderr)
        return 3
    dtype = getattr(torch, args.dtype)

    print(f"kintera {kintera.__version__}; {CELLS} cells {T_LO:g}-{T_HI:g} K, cell {CELL} at "
          f"{T_CELL} K and {P_CELL:g} Pa; {args.dtype}, max-iter {MAX_ITER}, ftol {args.ftol:g}; "
          f"compared at atol {ATOL:g}, rtol {RTOL:g}")
    try:
        cpu_out, cpu_gain, cpu_diag = run(torch, kintera, "cpu", args.ftol, dtype)
        gpu_out, gpu_gain, gpu_diag = run(torch, kintera, "cuda", args.ftol, dtype)
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        print(f"ftol_sweep.py: {type(e).__name__}: {str(e).splitlines()[0][:200]}", file=sys.stderr)
        return 1

    if cpu_diag is not None and gpu_diag is not None:
        print(f"  diagnostics: CPU {cpu_diag.flatten()[:4].tolist()}  "
              f"CUDA {gpu_diag.flatten()[:4].tolist()}")
    print("  " + compare(torch, cpu_out, gpu_out, "mole fractions"))
    print("  " + compare(torch, cpu_gain, gpu_gain, "gain entries"))
    print("  one pair of builds, one state: not a sweep over states or devices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
