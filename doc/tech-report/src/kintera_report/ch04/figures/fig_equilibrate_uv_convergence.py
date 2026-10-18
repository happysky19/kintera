"""Convergence of the fixed-energy, fixed-volume saturation adjustment (KKT branch) for one water reaction: the
residual |r| (panel a, logarithmic) and the temperature (panel b) per outer iteration, with the energy error of
every iterate at round-off.

Data: ../data/equilibrate_uv_history.json, written by checks/equilibrate_uv_check.py (a numpy port of
equilibrate_uv at the code pin of doc/tech-report/code_pin.yml, 0d9f67f97d841147e23283d2dcdab302238d2c45).
"""
import json
from pathlib import Path

import numpy as np

from kintera_report import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data" / "equilibrate_uv_history.json"


def make_fig():
    d = json.loads(DATA.read_text())
    fig, axes = fs.new_figure("double", 2.5, 1, 2)
    it = [h[0] for h in d["iterations"]]
    r = [abs(h[2]) for h in d["iterations"]]
    T = [h[1] for h in d["iterations"]]
    dU = [abs(h[3]) for h in d["iterations"]]
    ax = axes[0]
    ax.semilogy(it, np.maximum(r, 1e-16), marker="o", color=fs.GREEN, label="$|r|$ (log quotient)")
    ax.semilogy(it, np.maximum(dU, 1e-16), marker="s", color=fs.BLUE, ls=":", label="$|U - U_0|$ [J m$^{-3}$]")
    ax.axhline(1e-6, color=fs.VERMILLION, ls=":", lw=0.9)
    ax.text(it[-1], 1.6e-6, "default `ftol`", ha="right", va="bottom", fontsize=fs.SMALL_PT, color=fs.VERMILLION)
    ax.set_xlabel("outer iteration [-]")
    ax.set_ylabel("residual [-]")
    ax.set_xticks(it)
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(it, T, marker="o", color=fs.GREEN)
    ax.set_xlabel("outer iteration [-]")
    ax.set_ylabel("$T$ [K]")
    ax.set_xticks(it)
    fs.panel_labels(axes)
    return fig
