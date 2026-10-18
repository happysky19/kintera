"""Entropy of saturated water vapour and of the liquid that eval_entropy_R derives from it, against temperature, for
the h2o_ideal curve and the constant heat capacities of tests/jupiter.yaml (shifted offsets u_v* = -750 K,
u_c* = -6130 K, s_v* = -3.5 ln 300 + ln 1e5, Tref 300 K, Pref 1e5 Pa). Panel (a): s/R of both phases; panel (b):
the gap (s_v - s_c) R T against the enthalpy difference h_v - h_c, which the construction makes equal.

Evaluated here from the formulas of eval_entropy_R, eval_enthalpy_R and h2o_ideal at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
import numpy as np

from kintera_report import figstyle as fs

RGAS = 8.31446
TREF, PREF = 300.0, 1.0e5


def logsvp(T):
    betal, gammal, betas, gammas, tr, pr = 24.845, 4.986009, 22.98, 0.52, 273.16, 611.7
    t = T / tr
    liquid = (1 - 1 / t) * betal - gammal * np.log(t)
    solid = (1 - 1 / t) * betas - gammas * np.log(t)
    return np.where(T > tr, liquid, solid) + np.log(pr)


def make_fig():
    fig, axes = fs.new_figure("double", 2.5, 1, 2)
    T = np.linspace(230.0, 330.0, 201)
    L = logsvp(T)
    uv, uc = -2.5 * TREF, -3430.0 - 9.0 * TREF
    hv = uv + 2.5 * T + T
    hc = uc + 9.0 * T
    sv_star = -3.5 * np.log(TREF) + np.log(PREF)
    s_v = sv_star + 3.5 * np.log(T) - L          # vapour at its saturation partial pressure
    s_c = (hc - hv) / T + s_v
    ax = axes[0]
    ax.plot(T, s_v, color=fs.BLUE, label="saturated vapour $s_v/\\mathcal{R}$")
    ax.plot(T, s_c, color=fs.GREEN, ls="--", label="liquid $s_c/\\mathcal{R}$ (derived)")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$s/\\mathcal{R}$ [-]")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(T, (s_v - s_c) * RGAS * T / 1e3, color=fs.GREEN, label="$(s_v - s_c)\\,T$")
    ax.plot(T, (hv - hc) * RGAS / 1e3, color=fs.BLUE, ls=":", marker="o", markevery=25, label="$h_v - h_c$")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("[kJ mol$^{-1}$]")
    ax.legend(loc="upper right")
    fs.panel_labels(axes)
    return fig
