"""The energy residual F(T) = U(T) - U_0 of the partitioned state against temperature for one water pair, with its
kink at the temperature where the whole inventory fits in the vapour, and the bracket-and-Newton iterates of the
partition solver.

Data: ../data/uv_partition_energy.json, written by checks/uv_partition_check.py (a numpy port of
equilibrate_uv_partition at the code pin of doc/tech-report/code_pin.yml, 0d9f67f97d841147e23283d2dcdab302238d2c45).
"""
import json
from pathlib import Path

from kintera_report import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data" / "uv_partition_energy.json"


def make_fig():
    d = json.loads(DATA.read_text())
    fig, ax = fs.new_figure("double", 2.6)
    T = d["curve"]["T_K"]
    F = [f / 1e3 for f in d["curve"]["F_J_m3"]]
    ax.plot(T, F, color=fs.BLUE, label="$F(T) = U(T) - U_0$ of the partitioned state")
    ax.axhline(0.0, color=fs.BLACK, lw=0.8)
    its = d["iterates"]
    ax.plot([t for t, _ in its], [r / 1e3 for _, r in its], ls="none", marker="o", color=fs.GREEN, zorder=5,
            label="iterates (start, then Newton inside the bracket)")
    # the first three iterates are far apart; the rest sit on the solution and share one label
    for k, (t, r) in enumerate(its[:3]):
        ax.annotate(str(k), (t, r / 1e3), textcoords="offset points", xytext=(6, 6), fontsize=fs.SMALL_PT,
                    color=fs.GREEN)
    t, r = its[-1]
    ax.annotate(f"3 to {len(its) - 1}", (t, r / 1e3), textcoords="offset points", xytext=(8, -14),
                fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.axvline(d["T_solution_K"], color=fs.GREY, lw=0.8, ls=":")
    ax.text(d["T_solution_K"] + 0.8, min(F) * 0.9, f"solution {d['T_solution_K']:.2f} K", ha="left", va="bottom",
            fontsize=fs.SMALL_PT, color=fs.GREY)
    ax.set_xlabel("trial temperature $T$ [K]")
    ax.set_ylabel("$F(T)$ [kJ m$^{-3}$]")
    ax.legend(loc="upper left")
    return fig
