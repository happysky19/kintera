"""One `evolve_implicit` device/dtype/shape contract case per process.

    python -P src/kintera_report/ch12/runs/ei_case.py cuda ok
    python -P src/kintera_report/ch12/runs/ei_case.py cuda cpu
    python -P src/kintera_report/ch12/runs/ei_case.py cuda float32
    python -P src/kintera_report/ch12/runs/ei_case.py cuda narrow

One case per process on purpose: an illegal access poisons the CUDA context, so a second case in the
same interpreter would report the first one's damage.

The cases are the four the chapter cites:

  cuda ok       every argument on CUDA. On a -DCUDA=OFF build this is the "CUDA inputs to a CPU-only
                build stop" limit (`_dispatch.qmd`, `^ch12-dispatch-missing`): the stub has no CUDA
                entry and the call is expected to raise `DispatchStub: missing kernel for cuda`.
                On a -DCUDA=ON build the same case is expected to return.
  cuda cpu      rates and Jacobian on CUDA, `stoich` left on the CPU.
  cuda float32  every argument on CUDA, single precision.
  cuda narrow   every argument on CUDA, one reaction column of the deck's two.

The last three are the "mixed arguments" limit (`^ch12-dispatch-stoich`).

`stoich` comes from the deck, 64 cells as the chapter states. The rates and the Jacobian are
synthetic and seeded: this case is about the device, dtype and shape contract, not about a physical
trajectory, and a fixed seed makes the report reproducible. The deck is the committed copy of the
`tests/test_coagulation_pressure_switch.py` input in `derivation-checks/ch12/saved-preparation/`.
"""
import argparse
import os
import sys

CELLS = 64
SEED = 20261020
CASES = ("ok", "cpu", "float32", "narrow")
DECK = "test_coagulation_pressure_switch.yaml"


def _deck() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    saved = os.environ.get("CH12_SAVED") or os.path.join(
        here, "..", "..", "..", "..", "derivation-checks", "ch12", "saved-preparation")
    path = os.path.normpath(os.path.join(saved, DECK))
    if not os.path.exists(path):
        sys.exit(f"ei_case.py: no deck at {path} (set CH12_SAVED)")
    return path


def inputs(torch, kintera, dtype):
    """(rate, stoich, jacobian) for `CELLS` cells, from the deck's stoichiometry."""
    opts = kintera.KineticsOptions.from_yaml(_deck())
    kin = kintera.Kinetics(opts)
    stoich = kin.buffer("stoich").to(dtype)
    nspecies, nreactions = stoich.shape[0], stoich.shape[1]
    g = torch.Generator().manual_seed(SEED)
    rate = torch.rand((CELLS, nreactions), generator=g, dtype=torch.float64).to(dtype)
    jac = torch.zeros((CELLS, nreactions, nreactions), dtype=torch.float64)
    jac[:, range(nreactions), range(nreactions)] = 1.0
    return rate, stoich, jac.to(dtype), nspecies, nreactions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("device", choices=["cuda"], help="the device under test")
    ap.add_argument("case", choices=CASES)
    ap.add_argument("--dt", type=float, default=1.0)
    args = ap.parse_args()

    try:
        import torch
        import kintera as kt
    except ImportError as e:
        sys.exit(f"ei_case.py: {e}. Build and install kintera first.")

    if not torch.cuda.is_available():
        # Not a pass. The case is about CUDA; without a device there is no result to report.
        print(f"{args.case}: SKIPPED, no CUDA device visible", file=sys.stderr)
        return 3

    dtype = torch.float32 if args.case == "float32" else torch.float64
    rate, stoich, jac, nspecies, nreactions = inputs(torch, kt, dtype)

    if args.case == "narrow":
        if nreactions < 2:
            sys.exit(f"ei_case.py: deck has {nreactions} reaction(s); 'narrow' needs at least two")
        stoich = stoich[:, :1].contiguous()

    where = {"ok": ("cuda", "cuda", "cuda"),
             "cpu": ("cuda", "cpu", "cuda"),
             "float32": ("cuda", "cuda", "cuda"),
             "narrow": ("cuda", "cuda", "cuda")}[args.case]
    rate, stoich, jac = (t.to(d) for t, d in zip((rate, stoich, jac), where))

    print(f"kintera {kt.__version__}; case {args.case}; {CELLS} cells, "
          f"{nspecies} species x {nreactions} reactions; dtype {dtype}; "
          f"rate={rate.device} stoich={stoich.device} jacobian={jac.device}; "
          f"stoich shape {tuple(stoich.shape)}")
    try:
        out = kt.evolve_implicit(rate, stoich, jac, args.dt)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - the exception IS the measurement here
        print(f"{args.case}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return 1
    finite = bool(out.isfinite().all())
    print(f"{args.case}: returned {tuple(out.shape)} on {out.device}, dtype {out.dtype}, "
          f"all finite: {finite}")
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
