"""Layout of the two state vectors of ThermoX (mole fractions, species last) and ThermoY (mass fractions, species
first, reference species dropped) for the species of tests/jupiter.yaml, and the conversions between them.

A cartoon drawn from ThermoXImpl::compute and ThermoYImpl::compute at the code pin of doc/tech-report/code_pin.yml
(0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
from matplotlib.patches import FancyArrowPatch

from kintera_report import figstyle as fs

GASES = ["dry", "H$_2$O", "NH$_3$", "H$_2$S"]
CLOUDS = ["H$_2$O(l)", "NH$_3$(s)", "NH$_4$SH(s)"]
W, H = 0.78, 0.5


def _row(ax, x0, y, names, kinds):
    for k, (name, kind) in enumerate(zip(names, kinds)):
        fs.cell(ax, x0 + k * W, y, W, H, fill=(kind == "gas"))
        ax.text(x0 + (k + 0.5) * W, y + 0.5 * H, name, ha="center", va="center", fontsize=fs.SMALL_PT)


def make_fig():
    fig, ax = fs.new_figure("double", 2.6)
    fs.cartoon(ax)
    names = GASES + CLOUDS
    kinds = ["gas"] * 4 + ["cloud"] * 3
    # ThermoX: x_0 .. x_6 on the last axis
    y_top = 2.0
    _row(ax, 0.0, y_top, names, kinds)
    ax.text(0.0, y_top + H + 0.12, "ThermoX: $x_i$, shape (..., $1+n_y$), species on the last axis",
            ha="left", va="bottom", fontsize=fs.SMALL_PT)
    ax.text(-0.1, y_top - 0.12, "$i$ =", ha="right", va="top", fontsize=fs.SMALL_PT)
    for k in range(7):
        ax.text((k + 0.5) * W, y_top - 0.12, f"{k}", ha="center", va="top", fontsize=fs.SMALL_PT)
    # ThermoY: y_1 .. y_6 on the first axis, the reference species implied
    y_bot = 0.4
    _row(ax, W, y_bot, names[1:], kinds[1:])
    fs.cell(ax, 0.0, y_bot, W, H, kind="ghost")
    ax.text(0.5 * W, y_bot + 0.5 * H, "$1-\\sum y$", ha="center", va="center", fontsize=fs.SMALL_PT,
            bbox=dict(boxstyle="round,pad=0.1", fc=fs.WHITE, ec="none"))
    ax.text(0.0, y_bot - 0.12, "ThermoY: $y_i$, shape ($n_y$, ...), species on the first axis; the reference species is implied",
            ha="left", va="top", fontsize=fs.SMALL_PT)
    # arrows between the rows
    for x, label, up in ((2.0 * W, "Y->X", True), (4.0 * W, "X->Y", False)):
        p0, p1 = ((x, y_bot + H + 0.05), (x, y_top - 0.38)) if up else ((x, y_top - 0.38), (x, y_bot + H + 0.05))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=fs.GREEN))
        ax.text(x + 0.08, 0.5 * (y_bot + H + y_top - 0.38), label, ha="left", va="center", fontsize=fs.SMALL_PT,
                family=fs.MONO)
    ax.text(7 * W + 0.3, 0.5 * (y_bot + y_top + H), "filled: gas, the first $N_{\\mathrm{g}}$ entries\nwhite: condensate\nhatched: implied",
            ha="left", va="center", fontsize=fs.SMALL_PT)
    ax.set_xlim(-0.4, 8.6)
    ax.set_ylim(-0.1, 3.0)
    return fig
