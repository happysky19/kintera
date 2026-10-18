"""Molar internal energy of water vapour and liquid water with the constant heat capacities and offsets of
tests/jupiter.yaml (cv_R 2.5 and 9.0; u0_R 0 and -3430 K at Tref = 300 K): the two lines, the offsets as the card
gives them at Tref and as the code stores them at T = 0, and the latent heat as the vertical gap.

Evaluated here from the formulas of eval_intEng_R and ThermoYImpl::reset at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
import numpy as np

from kintera_report import figstyle as fs

TREF = 300.0
CV_V, U0_V = 2.5, 0.0
CV_C, U0_C = 9.0, -3430.0


def make_fig():
    fig, ax = fs.new_figure("double", 2.6)
    T = np.linspace(0.0, 400.0, 201)
    uv = U0_V + CV_V * (T - TREF)
    uc = U0_C + CV_C * (T - TREF)
    ax.plot(T, uv, color=fs.BLUE, label="vapour H$_2$O: $u/\\mathcal{R} = u^\\star_v/\\mathcal{R} + 2.5\\,T$")
    ax.plot(T, uc, color=fs.GREEN, ls="--", label="liquid H$_2$O(l): $u/\\mathcal{R} = u^\\star_c/\\mathcal{R} + 9.0\\,T$")
    # offsets at Tref as given by the card
    ax.plot([TREF, TREF], [U0_C, U0_V], color=fs.VERMILLION, lw=1.2)
    ax.plot([TREF], [U0_V], marker="o", color=fs.VERMILLION, ls="none")
    ax.plot([TREF], [U0_C], marker="o", color=fs.VERMILLION, ls="none")
    ax.text(TREF + 8, 0.5 * (U0_V + U0_C), "latent heat at $T_\\star$:\n$\\ell/\\mathcal{R}$ = 3430 K", ha="left",
            va="center", fontsize=fs.SMALL_PT, color=fs.VERMILLION)
    # stored offsets at T = 0
    ax.plot([0.0], [U0_V - CV_V * TREF], marker="s", color=fs.BLUE, ls="none")
    ax.plot([0.0], [U0_C - CV_C * TREF], marker="s", color=fs.GREEN, ls="none")
    ax.text(6.0, U0_V - CV_V * TREF + 150, "$u^\\star_v/\\mathcal{R}$ = -750 K", ha="left", va="bottom",
            fontsize=fs.SMALL_PT, color=fs.BLUE)
    ax.text(6.0, U0_C - CV_C * TREF - 150, "$u^\\star_c/\\mathcal{R}$ = -6130 K", ha="left", va="top",
            fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.axvline(TREF, color=fs.GREY, lw=0.8)
    ax.text(TREF + 6, -5200, "$T_\\star$ = 300 K (`Tref`)", ha="left", va="center", fontsize=fs.SMALL_PT, color=fs.GREY)
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$u/\\mathcal{R}$ [K]")
    ax.set_xlim(0, 400)
    ax.set_ylim(-7500, 1500)
    ax.legend(loc="lower right")
    return fig
