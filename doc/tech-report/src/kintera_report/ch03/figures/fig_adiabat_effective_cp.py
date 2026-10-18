"""The effective heat capacity of a water-saturated parcel of dry air at 1 bar against temperature: the frozen part
R sum x_i cp_i/R and the latent part l L'(T) x_v x_g/(x_g - x_v) of effective_cp for one condensing vapour, with the
h2o_ideal curve and the constant heat capacities of tests/jupiter.yaml; panel (b) the moist and dry adiabatic
lapse rates dln T/dln p = R x_g / c_p that extrapolate_dlnp integrates.

Evaluated here from the single-reaction closed form checked in checks/adiabat_check.py ([C2]) at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
import numpy as np

from kintera_report import figstyle as fs

RGAS = 8.31446


def make_fig():
    fig, axes = fs.new_figure("double", 2.5, 1, 2)
    T = np.linspace(240.0, 320.0, 161)
    p = 1.0e5
    betal, gammal, betas, gammas, tr, pr = 24.845, 4.986009, 22.98, 0.52, 273.16, 611.7
    t = T / tr
    L = np.where(T > tr, (1 - 1 / t) * betal - gammal * np.log(t), (1 - 1 / t) * betas - gammas * np.log(t)) + np.log(pr)
    Lp = np.where(T > tr, betal * tr / T**2 - gammal / T, betas * tr / T**2 - gammas / T)
    xv = np.minimum(np.exp(L) / p, 0.5)
    xg = 1.0
    xd = xg - xv
    cp_normal = RGAS * (xd * 3.5 + xv * 3.5)
    hv = (-750.0 + 3.5 * T) * RGAS
    hc = (-6130.0 + 9.0 * T) * RGAS
    cp_latent = (hv - hc) * Lp * xv * xg / (xg - xv)
    ax = axes[0]
    ax.plot(T, cp_normal, color=fs.BLACK, ls=":", label="frozen $\\sum_i x_i c_{p,i}$")
    ax.plot(T, cp_normal + cp_latent, color=fs.GREEN, label="effective $c_p$ (with the latent term)")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$c_p$ [J mol$^{-1}$ K$^{-1}$]")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(T, RGAS * xg / cp_normal, color=fs.BLACK, ls=":", label="dry")
    ax.plot(T, RGAS * xg / (cp_normal + cp_latent), color=fs.GREEN, label="saturated")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$\\mathrm{d}\\ln T/\\mathrm{d}\\ln p$ [-]")
    ax.legend(loc="lower left")
    fs.panel_labels(axes)
    return fig
