"""Heat capacity of H2 under the three caloric models of eval_uhs.cpp: the constant baseline cv_R + 1 = 3.5, the
NASA-9 polynomial (`use-nasa9-cp`) and the rigid-rotor partition function (`use-h2-cp`) in its equilibrium and
normal ortho-para modes.

Data: ../data/nasa9_h2.dat (the H2 record of data/nasa9.dat at the code pin of doc/tech-report/code_pin.yml,
0d9f67f97d841147e23283d2dcdab302238d2c45); the rotational curves are evaluated here with the formulas of
eval_h2cp (theta_rot = 87.55 K, J <= 40).
"""
from pathlib import Path

import numpy as np

from kintera_report import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data" / "nasa9_h2.dat"
THETA_ROT, JMAX = 87.55, 40


def nasa9_cp_R(T):
    lines = [l for l in DATA.read_text().splitlines() if l and not l.startswith("#")]
    v = np.array(" ".join(lines[1:5]).split(), dtype=float)
    a = np.where((T < 1000.0)[:, None], v[None, 0:7], v[None, 10:17])
    return (a * T[:, None] ** np.arange(-2, 5)).sum(-1)


def h2_cp_R(T, mode):
    J = np.arange(0, JMAX + 1, dtype=float)
    m, deg, odd = J * (J + 1), 2 * J + 1, (J % 2) == 1

    def cv_rot(w):
        wx = w[None, :] * np.exp(-THETA_ROT * m[None, :] / T[:, None])
        Z = wx.sum(-1)
        m1 = (wx * m).sum(-1) / Z
        m2 = (wx * m * m).sum(-1) / Z
        return (THETA_ROT / T) ** 2 * (m2 - m1 * m1)

    if mode == "normal":
        return 2.5 + 0.25 * cv_rot(deg * (~odd)) + 0.75 * cv_rot(deg * odd)
    return 2.5 + cv_rot(deg * np.where(odd, 3.0, 1.0))


def make_fig():
    fig, ax = fs.new_figure("double", 2.6)
    T = np.geomspace(20.0, 3000.0, 400)
    ax.plot(T, np.full_like(T, 3.5), color=fs.BLACK, ls=":", label="constant, $c_v/\\mathcal{R}+1$ = 3.5")
    Tn = T[T >= 200.0]
    ax.plot(Tn, nasa9_cp_R(Tn), color=fs.BLUE, ls="-", label="NASA-9 (`use-nasa9-cp`), 200-3000 K")
    ax.plot(T, h2_cp_R(T, "equilibrium"), color=fs.GREEN, ls="-", marker="o", markevery=40,
            label="rotor, equilibrium ortho-para (`use-h2-cp`)")
    ax.plot(T, h2_cp_R(T, "normal"), color=fs.ORANGE, ls="--", marker="^", markevery=40,
            label="rotor, normal 1:3 (`h2-cp-mode: normal`)")
    ax.set_xscale("log")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$c_p/\\mathcal{R}$ [-]")
    ax.set_ylim(2.2, 4.9)
    ax.axvline(300.0, color=fs.GREY, lw=0.8)
    ax.text(310.0, 2.3, "$T_0$ = 300 K", ha="left", va="center", fontsize=fs.SMALL_PT, color=fs.GREY)
    ax.legend(loc="upper right", ncol=1)
    return fig
