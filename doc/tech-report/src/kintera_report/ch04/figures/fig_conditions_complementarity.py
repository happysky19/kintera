"""The admissible set of a phase-change reaction in the plane of its saturation residual and its condensate amount,
and the four decisions of the active-set test of the saturation solvers.

A cartoon drawn from the active-set test of equilibrate_tp and equilibrate_uv at the code pin of
doc/tech-report/code_pin.yml (0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
from matplotlib.patches import FancyArrowPatch, Rectangle

from kintera_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure("double", 2.6)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.12, 1.05)
    # the admissible set: the half-line r = 0, x_c >= 0 and the segment r <= 0, x_c = 0
    ax.plot([0, 0], [0, 1.0], color=fs.BLUE, lw=2.5, solid_capstyle="butt", zorder=4)
    ax.plot([-1.6, 0], [0, 0], color=fs.BLUE, lw=2.5, solid_capstyle="butt", zorder=4)
    ax.plot([0], [0], marker="o", color=fs.BLUE, ms=6, zorder=5)
    # tolerance band
    ax.add_patch(Rectangle((-0.06, 0), 0.12, 1.0, fc=fs.CELL_FILL, ec="none", zorder=1))
    ax.text(0.08, 0.95, "$|r_j| \\leq \\varepsilon_{\\mathrm{tol}}$:\nin equilibrium", ha="left", va="top",
            fontsize=fs.SMALL_PT, color=fs.BLUE)
    # regions and their decisions
    ax.text(0.85, 0.55, "$r_j > 0$: supersaturated\ncondensation, active\nwith or without cloud", ha="center",
            va="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.text(-0.85, 0.6, "$r_j < 0$, $x_c > 0$:\nevaporation, active", ha="center", va="center", fontsize=fs.SMALL_PT,
            color=fs.GREEN)
    ax.text(-0.85, -0.07, "$r_j < 0$, $x_c = 0$: nothing to evaporate, in equilibrium", ha="center", va="top",
            fontsize=fs.SMALL_PT, color=fs.BLUE)
    for (x0, y0, x1, y1) in [(0.85, 0.3, 0.1, 0.3), (-0.85, 0.42, -0.06, 0.42)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=1.2, color=fs.GREEN,
                                     zorder=3))
    ax.text(0.47, 0.33, "step toward $r_j = 0$", ha="center", va="bottom", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.set_xlabel("saturation residual $r_j = \\ln Q_j - L_j(T)$ [-]")
    ax.set_ylabel("condensate $x_c$ or $c_c$")
    ax.set_yticks([0])
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["$<0$", "0", "$>0$"])
    return fig
