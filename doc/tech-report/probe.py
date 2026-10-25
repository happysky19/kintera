"""ln K of named func1 entries at 250 K and 300 K, on CPU and on CUDA tensors.

The scratch script of `^ch12-devfn-scratch`, committed at the path that footnote names, so that
`python -P probe.py <names>` resolves from `doc/tech-report` instead of exiting 2.

NOT recovered: the footnote's own script was "patched in a scratch copy only, not committed" and is
not in the report's history. The recovered producer at commit 5169175 already owns the name
probe.py there and is a different experiment (a CPU/CUDA comparison campaign, not a device-table
mutation). This file is the committed stand-in, written to the footnote's recipe.

    python -P probe.py h2o_ideal h2o_ideal_ddT nh3_ideal na_h2s_visscher_ddT

This is the probe behind the chapter-12 limit "a function missing from the device table gives wrong
values on CUDA, not an error" (`_device-functions.qmd`, `^ch12-devfn-scratch`). It is only half the
experiment: the other half is `func_table_gap.py` next to this file, which adds a test function to
the CPU table and `func1_names` without touching the CUDA table, so that the two tables disagree,
and takes the change back afterwards. Applied, rebuilt CPU-side only and probed with this script, the CPU and CUDA columns below
part company; unapplied, this script is a plain CPU-versus-CUDA agreement check and should print
`same` on every row.

ln K comes from `kintera.relative_humidity` the way `derivation-checks/check_ch2_svp.py` takes it:
every vapour at 1 Pa, so RH_j = exp(-ln K_j). Nothing here is specific to a GPU model; the device
table index is.

Needs `KINTERA_REPO` pointing at a kintera checkout, for the deck.
"""
import argparse
import os
import sys

R = 8.31446261815324

DECK = ("tests", "jupiter.yaml")


def _deck() -> str:
    repo = os.environ.get("KINTERA_REPO")
    if not repo:
        sys.exit("func_table_probe.py: set KINTERA_REPO to a kintera checkout (it supplies the deck)")
    path = os.path.join(repo, *DECK)
    if not os.path.exists(path):
        sys.exit(f"func_table_probe.py: no deck at {path}")
    return path


def log_k(kintera, torch, op, th, names, temps):
    """ln K_j(T) for every nucleation reaction, as check_ch2_svp.py computes it."""
    ngas = len(op.vapor_ids())
    T = torch.tensor(temps, dtype=torch.float64)
    c = torch.zeros(len(temps), len(op.species()), dtype=torch.float64)
    c[:, 1:ngas] = (1.0 / (R * T)).unsqueeze(-1)
    c[:, 0] = 1.0
    out = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            out[device] = None
            continue
        rh = kintera.relative_humidity(T.to(device), c.to(device),
                                       th.buffer("stoich").to(device), op.nucleation(), ngas)
        out[device] = (-rh.log()).cpu()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="func1 names to report; default: every nucleation formula")
    ap.add_argument("--temps", default="250,300", help="kelvin, comma separated")
    ap.add_argument("--self-test", action="store_true",
                    help="check the argument handling and ln K construction without a GPU")
    args = ap.parse_args()
    temps = [float(t) for t in args.temps.split(",")]

    try:
        import torch
        import kintera
    except ImportError as e:
        sys.exit(f"func_table_probe.py: {e}. Build and install kintera first.")

    op = kintera.ThermoOptions.from_yaml(_deck())
    th = kintera.ThermoX(op)
    forms = list(op.nucleation().logsvp())
    wanted = args.names or forms
    missing = [n for n in wanted if n not in forms]
    if missing:
        # Never silently report nothing: a name the deck does not register is the user's answer.
        print(f"not registered by this deck: {', '.join(missing)}", file=sys.stderr)
        print(f"deck registers: {', '.join(forms)}", file=sys.stderr)
        if not [n for n in wanted if n in forms]:
            return 2

    cols = log_k(kintera, torch, op, th, wanted, temps)
    if cols["cuda"] is None and not args.self_test:
        print("no CUDA device visible: the CPU column alone cannot show a device-table gap",
              file=sys.stderr)

    print(f"kintera {kintera.__version__}; ln K at {', '.join(f'{t:g} K' for t in temps)}")
    print(f"{'name':<24}{'T':>8}{'cpu':>16}{'cuda':>16}   verdict")
    worst = 0.0
    for name in wanted:
        if name not in forms:
            continue
        j = forms.index(name)
        for i, t in enumerate(temps):
            cpu = cols["cpu"][i, j].item()
            if cols["cuda"] is None:
                print(f"{name:<24}{t:>8g}{cpu:>16.6f}{'n/a':>16}   cpu only")
                continue
            gpu = cols["cuda"][i, j].item()
            d = abs(cpu - gpu)
            worst = max(worst, d)
            print(f"{name:<24}{t:>8g}{cpu:>16.6f}{gpu:>16.6f}   "
                  f"{'same' if d == 0 else f'DIFFER by {d:.6g}'}")
    if cols["cuda"] is not None:
        print(f"largest CPU/CUDA difference: {worst:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
