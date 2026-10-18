"""Dependency map of chapter 3 (STYLE 10.1): the seven schemes as nodes, "requires" as arrows.

Drawn from the call structure of src/thermo at the code pin of doc/tech-report/code_pin.yml
(0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
from matplotlib.patches import FancyArrowPatch

from kintera_report import figstyle as fs

NODES = {
    "species-state": (0.0, 1.0, "species and\nstate vectors"),
    "eos": (1.6, 1.0, "equation\nof state"),
    "caloric": (3.2, 1.0, "heat capacities\nand energy"),
    "reference-state": (3.2, 0.0, "reference state\nand latent heat"),
    "entropy": (4.8, 1.0, "entropy and\nfree energies"),
    "relative-humidity": (1.6, 0.0, "saturation\nratio"),
    "adiabat": (6.4, 1.0, "moist adiabat,\neffective $c_p$"),
}
# (from, to): "to" requires "from"
EDGES = [("species-state", "eos"), ("eos", "caloric"), ("reference-state", "caloric"), ("caloric", "entropy"),
         ("reference-state", "entropy"), ("entropy", "adiabat"), ("species-state", "relative-humidity"),
         ("caloric", "adiabat")]
W, H = 1.25, 0.55


def make_fig():
    fig, ax = fs.new_figure("double", 1.9)
    fs.cartoon(ax)
    for key, (x, y, label) in NODES.items():
        fs.cell(ax, x - W / 2, y - H / 2, W, H, fill=True)
        ax.text(x, y, label, ha="center", va="center", fontsize=fs.SMALL_PT)
    for a, b in EDGES:
        xa, ya, _ = NODES[a]
        xb, yb, _ = NODES[b]
        if ya == yb:
            p0, p1 = (xa + W / 2, ya), (xb - W / 2, yb)
        elif xa == xb:
            p0, p1 = (xa, ya + H / 2), (xb, yb - H / 2)
        else:
            p0, p1 = (xa + W / 2, ya + 0.1), (xb - W / 2 + 0.1, yb - H / 2)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=fs.GREEN,
                                     shrinkA=1, shrinkB=1, zorder=5))
    ax.text(6.4, 0.0, "arrow: requires", ha="center", va="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.set_xlim(-0.8, 7.2)
    ax.set_ylim(-0.45, 1.45)
    return fig
