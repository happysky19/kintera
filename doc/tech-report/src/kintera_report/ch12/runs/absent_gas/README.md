# The absent-gas structural survey (recovered, provenance incomplete)

These are the authentic scripts behind `^ch12-dispatch-absent`, recovered from a retained local
bundle. **They have no producing git SHA**: the bundle is not a git checkout, so unlike
`probe.py` and `one_cell.py` — which were restored from the report's own history at
`516917515670447774cada7d52ba5cb705df4987` — there is no commit to cite for these.

    K15_LEG=pin K15_WT=<kintera checkout at code_pin.yml> python run_sweep.py

`run_sweep.py` fixes the grid: 17 temperatures from 110 to 1600 K, 6 pressures from 1e2 to 1e7 Pa,
5 vapour amounts, over the decks `jupiter`, `earth`, `jup-thermo` and `kcl_mgsio3`. `sweep.py`
does the structural scan; `deriv.py` and `run_deriv.py` are the **selected** finite-difference
controls, which are a different and much smaller population than the survey.

## What the retained run actually covered

The retained output (`fix_sweep.txt`) totals **12750 cells, 14790 absent-gas reaction instances,
14688 at a converged exit, 0 cloud violations, 0 non-zero gain rows** — the chapter's numbers.
Per deck:

| deck | cells | absent-gas instances | converged |
|---|---|---|---|
| jupiter | 3570 | 5610 | 5610 |
| earth | 1020 | 510 | 510 |
| kcl_mgsio3 | 8160 | 8670 | **8568** |
| jup-thermo | — | — | SKIPPED |

Two limits the chapter does not state:

- **102 instances did not converge**, all in `kcl_mgsio3` (8670 − 8568). "14688 at a converged
  exit" is accurate but is not the whole population.
- `jup-thermo` was skipped with `AttributeError: 'NoneType' object has no attribute 'max_iter'`,
  not merely "could not be loaded".

## The provenance gap

The retained output is labelled `LEG fix`, i.e. a **fix** working tree, not the `pin` leg at
`code_pin.yml`. The chapter attributes these numbers to the pin. Running the `pin` leg here would
be **new evidence**, not a restoration of the historical pin output, which is not retained. Until
someone does that, the numbers should be read as coming from a build that is not the cited one.
