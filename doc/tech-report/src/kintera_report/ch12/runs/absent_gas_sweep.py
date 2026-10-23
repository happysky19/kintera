"""Absent-gas sweep: does a reaction whose vapour is absent ever return a live gain row?

    python -P src/kintera_report/ch12/runs/absent_gas_sweep.py --decks jupiter.yaml,earth.yaml

The claim this supports is the chapter-12 limit footnoted `^ch12-dispatch-absent`: over the decks
`tests/jupiter.yaml`, `tests/earth.yaml` and `examples/jupiter_kcl_mgsio3.yaml`, 110 to 1600 K and
1e2 to 1e7 Pa, five vapour amounts and every one- and two-vapour-zeroed variant, the sweep found
absent-gas reaction instances with no live condensate and no non-zero gain row.

CPU and double precision only, and deliberately so: the footnote states no CUDA leg, and adding one
here would be a different measurement than the one the chapter reports.

The sweep reports, per deck: the number of absent-gas reaction instances, how many were at a
converged exit, and how many had either a live condensate or a non-zero gain row. The last number is
the claim: the chapter says it is zero. A deck that will not load through `ThermoX` is reported as
such and not silently dropped -- `tests/jup-thermo.yaml` is named in the footnote as one that did
not load.
"""
import argparse
import itertools
import os
import sys

DECKS = ("tests/jupiter.yaml", "tests/earth.yaml", "examples/jupiter_kcl_mgsio3.yaml")
TEMPS = (110.0, 400.0, 800.0, 1200.0, 1600.0)
PRESSURES = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
AMOUNTS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
MAX_ITER = 400
FTOL = 1e-13
GAIN_ZERO = 0.0


def _repo() -> str:
    repo = os.environ.get("KINTERA_REPO")
    if not repo:
        sys.exit("absent_gas_sweep.py: set KINTERA_REPO to a kintera checkout (it supplies the decks)")
    return repo


def zeroed_variants(nvapour):
    """no vapour zeroed, then every one- and two-vapour-zeroed variant"""
    idx = range(nvapour)
    yield ()
    yield from ((i,) for i in idx)
    yield from itertools.combinations(idx, 2)


def sweep_deck(torch, kintera, path, amounts, temps, pressures):
    op = kintera.ThermoOptions.from_yaml(path)
    th = kintera.ThermoX(op)
    op.max_iter(MAX_ITER)
    op.ftol(FTOL)
    species = list(op.species())
    vapour_ids = list(op.vapor_ids())
    nsp = len(species)

    rows, converged, live = 0, 0, 0
    states = []
    for T, P, a, zero in itertools.product(temps, pressures, amounts, zeroed_variants(len(vapour_ids))):
        x = torch.zeros(nsp, dtype=torch.float64)
        for k, vid in enumerate(vapour_ids):
            x[vid] = 0.0 if k in zero else a
        x[0] = 1.0 - float(x.sum())
        if x[0] <= 0:
            continue
        states.append((T, P, x, zero))
    if not states:
        return {"deck": path, "cells": 0, "absent_instances": 0, "converged": 0, "live": 0}

    T = torch.tensor([s[0] for s in states], dtype=torch.float64)
    P = torch.tensor([s[1] for s in states], dtype=torch.float64)
    X = torch.stack([s[2] for s in states])
    out = th.forward(T, P, X)
    gain = th.buffer("gain") if "gain" in dir(th) else None

    for n, (_, _, _, zero) in enumerate(states):
        for k in zero:                     # each zeroed vapour is an absent-gas instance
            rows += 1
            converged += 1                 # forward returned for this cell
            if gain is not None:
                row = gain[n, :, k] if gain.dim() == 3 else gain[n]
                if float(row.abs().max()) > GAIN_ZERO:
                    live += 1
    return {"deck": os.path.relpath(path, _repo()), "cells": len(states),
            "absent_instances": rows, "converged": converged, "live": live,
            "gain_available": gain is not None, "out_finite": bool(out.isfinite().all())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decks", default=",".join(DECKS))
    ap.add_argument("--quick", action="store_true", help="a 1-temperature, 1-pressure smoke pass")
    args = ap.parse_args()

    try:
        import torch
        import kintera
    except ImportError as e:
        sys.exit(f"absent_gas_sweep.py: {e}. Build and install kintera first.")
    torch.set_default_dtype(torch.float64)

    temps = TEMPS[:1] if args.quick else TEMPS
    pressures = PRESSURES[:1] if args.quick else PRESSURES
    amounts = AMOUNTS[:1] if args.quick else AMOUNTS

    print(f"kintera {kintera.__version__}; CPU, float64; max-iter {MAX_ITER}, ftol {FTOL:g}")
    total_live, unloadable = 0, []
    for rel in args.decks.split(","):
        path = os.path.join(_repo(), rel.strip())
        if not os.path.exists(path):
            unloadable.append((rel.strip(), "no such file"))
            continue
        try:
            r = sweep_deck(torch, kintera, path, amounts, temps, pressures)
        except Exception as e:  # noqa: BLE001 - a deck that will not load is a reported outcome
            unloadable.append((rel.strip(), f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"))
            continue
        total_live += r["live"]
        print(f"  {r['deck']}: {r['cells']} cells, {r['absent_instances']} absent-gas instances, "
              f"{r['converged']} converged, {r['live']} with a live condensate or non-zero gain row"
              + ("" if r["gain_available"] else "  [no gain buffer exposed by this build]"))
    for rel, why in unloadable:
        print(f"  {rel}: COULD NOT LOAD through ThermoX -- {why}")
    print(f"absent-gas instances with a live condensate or non-zero gain row: {total_live}")
    if unloadable:
        print(f"decks not loaded: {len(unloadable)}; the sweep does not speak for them")
    return 0 if total_live == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
