"""The saturation ratio Q/Q_sat of relative_humidity against temperature for two reactions at fixed gas
concentrations: H2O <=> H2O(l) (one partial pressure, A = 1) and NH3 + H2S <=> NH4SH(s) (a product of two partial
pressures, A = 2), with the ratio 1 marked. Concentrations: 0.5 mol m^-3 of H2O; 0.02 mol m^-3 each of NH3 and H2S.

Evaluated here from the formulas of relative_humidity, h2o_ideal and nh3_h2s_lewis at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
import numpy as np

from kintera_report import figstyle as fs

RGAS = 8.31446


def h2o_ideal(T):
    betal, gammal, betas, gammas, tr, pr = 24.845, 4.986009, 22.98, 0.52, 273.16, 611.7
    t = T / tr
    return np.where(T > tr, (1 - 1 / t) * betal - gammal * np.log(t), (1 - 1 / t) * betas - gammas * np.log(t)) + np.log(pr)


def nh3_h2s_lewis(T):
    return (14.82 - 4705.0 / T) * np.log(10.0) + 2.0 * np.log(101325.0)


def make_fig():
    fig, ax = fs.new_figure("double", 2.5)
    T = np.linspace(150.0, 320.0, 341)
    c_h2o, c_nh3, c_h2s = 0.5, 0.02, 0.02
    rh_h2o = c_h2o / np.exp(h2o_ideal(T) - 1 * np.log(RGAS * T))
    rh_nh4sh = c_nh3 * c_h2s / np.exp(nh3_h2s_lewis(T) - 2 * np.log(RGAS * T))
    ax.plot(T, rh_h2o, color=fs.BLUE, label="H$_2$O $\\rightleftharpoons$ H$_2$O(l), $A_j$ = 1, 0.5 mol m$^{-3}$")
    ax.plot(T, rh_nh4sh, color=fs.GREEN, ls="--",
            label="NH$_3$ + H$_2$S $\\rightleftharpoons$ NH$_4$SH(s), $A_j$ = 2, 0.02 mol m$^{-3}$ each")
    ax.axhline(1.0, color=fs.VERMILLION, lw=0.9, ls=":")
    ax.text(152.0, 1.4, "saturated above, unsaturated below", ha="left", va="bottom", fontsize=fs.SMALL_PT,
            color=fs.VERMILLION)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e6)
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$Q_j/Q_j^{\\mathrm{sat}}$ [-]")
    ax.legend(loc="upper right")
    return fig
