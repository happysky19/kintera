"""Dependency map of chapter 4 (STYLE 10.1): the five schemes as nodes, "requires" as arrows, with the two
chapters it leans on (chapter 3 for the state and caloric functions, chapter 10 for the KKT least-squares core).

Drawn from the call structure of src/thermo at the code pin of doc/tech-report/code_pin.yml
(0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
from matplotlib.patches import FancyArrowPatch

from kintera_report import figstyle as fs

NODES = {
    "conditions": (0.0, 1.0, "saturation\nconditions"),
    "tp": (1.7, 1.0, "fixed $T$, $p$\nsolver"),
    "uv": (3.4, 1.0, "fixed $U$, $V$\nsolver"),
    "partition": (3.4, 0.0, "partition\nsolver"),
    "drivers": (5.1, 1.0, "batched\ndrivers"),
    "ch03": (0.0, 0.0, "state and caloric\nfunctions (ch. 3)"),
    "ch10": (1.7, 0.0, "KKT least squares\n(ch. 10)"),
}
EDGES = [("conditions", "tp"), ("tp", "uv"), ("uv", "drivers"), ("partition", "uv"), ("ch03", "conditions"),
         ("ch10", "tp"), ("tp", "drivers")]
W, H = 1.3, 0.55


def make_fig():
    fig, ax = fs.new_figure("double", 1.9)
    fs.cartoon(ax)
    for key, (x, y, label) in NODES.items():
        fs.cell(ax, x - W / 2, y - H / 2, W, H, fill=not key.startswith("ch"), kind="fluid")
        ax.text(x, y, label, ha="center", va="center", fontsize=fs.SMALL_PT,
                color=fs.GREY if key.startswith("ch") else fs.BLACK)
    for a, b in EDGES:
        xa, ya, _ = NODES[a]
        xb, yb, _ = NODES[b]
        if ya == yb:
            p0, p1 = (xa + W / 2, ya), (xb - W / 2, yb)
        elif xa == xb:
            p0, p1 = (xa, ya + H / 2), (xb, yb - H / 2)
        else:
            p0, p1 = (xa + W / 2, ya + 0.1), (xb - W / 2 + 0.15, yb - H / 2)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=fs.GREEN,
                                     shrinkA=1, shrinkB=1, zorder=5))
    ax.text(5.1, 0.0, "arrow: requires\ngrey: other chapters", ha="center", va="center", fontsize=fs.SMALL_PT,
            color=fs.GREEN)
    ax.set_xlim(-0.85, 5.95)
    ax.set_ylim(-0.45, 1.45)
    return fig
