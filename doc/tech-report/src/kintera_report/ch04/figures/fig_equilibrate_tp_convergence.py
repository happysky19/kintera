"""Convergence of the fixed-temperature, fixed-pressure saturation adjustment for one water reaction: the residual
|r| per iteration for a condensing start (panel a, logarithmic) and the mole fractions per iteration for the
evaporating start, where the extent limiter takes the cloud to zero in one step (panel b).

Data: ../data/equilibrate_tp_history.json, written by checks/equilibrate_tp_check.py (a numpy port of
equilibrate_tp at the code pin of doc/tech-report/code_pin.yml, 0d9f67f97d841147e23283d2dcdab302238d2c45).
"""
import json
from pathlib import Path

import numpy as np

from kintera_report import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data" / "equilibrate_tp_history.json"


def make_fig():
    d = json.loads(DATA.read_text())
    fig, axes = fs.new_figure("double", 2.5, 1, 2)
    ax = axes[0]
    it = [h[0] for h in d["condensation"]["iterations"]]
    r = [abs(h[1]) for h in d["condensation"]["iterations"]]
    ax.semilogy(it, np.maximum(r, 1e-16), marker="o", color=fs.GREEN, label="$|r|$, condensing start")
    ax.axhline(1e-6, color=fs.VERMILLION, ls=":", lw=0.9)
    ax.text(it[-1], 1.6e-6, "default `ftol`", ha="right", va="bottom", fontsize=fs.SMALL_PT, color=fs.VERMILLION)
    ax.set_xlabel("iteration [-]")
    ax.set_ylabel("$|r_j|$ [-]")
    ax.set_xticks(it)
    ax.legend(loc="upper right")
    ax = axes[1]
    it = [h[0] for h in d["evaporation"]["iterations"]]
    x = np.array([h[2] for h in d["evaporation"]["iterations"]])
    ax.plot(it, x[:, 1], marker="o", color=fs.BLUE, label="vapour $x_v$")
    ax.plot(it, x[:, 2], marker="^", color=fs.GREEN, ls="--", label="cloud $x_c$")
    ax.set_xlabel("iteration [-]")
    ax.set_ylabel("mole fraction [-]")
    ax.set_xticks(it)
    ax.set_ylim(-0.005, 0.06)
    ax.legend(loc="center right")
    fs.panel_labels(axes)
    return fig
