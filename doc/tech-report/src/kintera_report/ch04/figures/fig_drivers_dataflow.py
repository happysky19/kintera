"""Data flow of one ThermoX.forward or ThermoY.forward call: the tensors that enter, the per-cell kernel that the
tensor iterator dispatches to CPU or CUDA, and the outputs (adjusted state, gain, diagnostic, failure count), with
the warm-start test on the saved active set.

A cartoon drawn from ThermoXImpl::forward and ThermoYImpl::forward at the code pin of doc/tech-report/code_pin.yml
(0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
from matplotlib.patches import FancyArrowPatch

from kintera_report import figstyle as fs

W, H = 3.0, 0.56


def box(ax, x, y, label, fill=True, color=fs.BLACK):
    fs.cell(ax, x - W / 2, y - H / 2, W, H, fill=fill)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs.SMALL_PT, color=color)


def arrow(ax, p0, p1, color=fs.GREEN, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, ls=ls,
                                 shrinkA=1, shrinkB=1, zorder=5))


def make_fig():
    fig, ax = fs.new_figure("double", 2.7)
    fs.cartoon(ax)
    xin, xit, xk, xout = 0.0, 3.6, 7.0, 10.6
    # inputs
    box(ax, xin, 1.7, "$T$, $p$, $x$ (ThermoX)\n$\\rho$, $U$, $y$ (ThermoY)")
    box(ax, xin, 0.7, "options: stoich,\ncurves, ftol, max-iter")
    box(ax, xin, -0.4, "saved active set\n(reaction_set, nactive)", fill=False)
    # the iterator and kernel
    box(ax, xit, 1.2, "tensor iterator\nover the cells")
    box(ax, xk, 1.2, "per-cell kernel\nequilibrate_tp / _uv")
    ax.text(xk, 1.2 + H / 2 + 0.08, "CPU threads or CUDA blocks", ha="center", va="bottom", fontsize=fs.SMALL_PT,
            color=fs.GREY)
    # outputs
    box(ax, xout, 1.95, "state adjusted\nin place")
    box(ax, xout, 1.2, "gain $(\\dots, N_r, N_r)$")
    box(ax, xout, 0.45, "diag $(\\dots, 1)$")
    box(ax, xout, -0.4, "nfail += count(diag < 0)\n(ThermoY)", fill=False)
    arrow(ax, (xin + W / 2, 1.7), (xit - W / 2, 1.3))
    arrow(ax, (xin + W / 2, 0.7), (xit - W / 2, 1.1))
    arrow(ax, (xin + W / 2, -0.4), (xit - W / 2, 0.98), color=fs.VERMILLION, ls="--")
    ax.text(xin + W / 2 + 0.25, -0.55, "same shape\nand device?", ha="left", va="center", fontsize=fs.SMALL_PT,
            color=fs.VERMILLION)
    arrow(ax, (xit + W / 2, 1.2), (xk - W / 2, 1.2))
    for y in (1.95, 1.2, 0.45):
        arrow(ax, (xk + W / 2, 1.2 + (y - 1.2) * 0.4), (xout - W / 2, y))
    arrow(ax, (xout, 0.45 - H / 2), (xout, -0.4 + H / 2), color=fs.VERMILLION, ls="--")
    ax.set_xlim(xin - W / 2 - 0.1, xout + W / 2 + 0.1)
    ax.set_ylim(-0.8, 2.35)
    return fig
