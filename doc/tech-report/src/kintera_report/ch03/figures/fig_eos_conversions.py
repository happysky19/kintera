"""The conversion keys of ThermoY and ThermoX as a graph: state quantities as nodes, each `compute` key as an arrow
from the input it iterates on (or is named after) to its output; the keys that iterate (Newton loops) are dashed.

Drawn from ThermoYImpl::compute and ThermoXImpl::compute at the code pin of doc/tech-report/code_pin.yml
(0d9f67f97d841147e23283d2dcdab302238d2c45); no data file.
"""
from matplotlib.patches import Circle, FancyArrowPatch

from kintera_report import figstyle as fs

# panel (a): ThermoY, mass basis; panel (b): ThermoX, mole basis. Node: (x, y); edge: (from, to, key, newton, label dx, dy)
Y_NODES = {"D,Y": (0.0, 1.2), "V": (1.6, 1.2), "T": (3.4, 2.2), "P": (5.2, 2.2), "U": (3.4, 0.2), "S": (5.2, 0.2),
           "cv": (1.6, 0.0)}
Y_EDGES = [("D,Y", "V", "DY->V", False, 0.0, 0.14), ("V", "T", "PV->T", True, -0.25, 0.22),
           ("V", "P", "VT->P", False, 0.35, 0.14), ("V", "U", "VT->U", False, -0.25, -0.26),
           ("U", "T", "VU->T", True, 0.42, 0.0), ("V", "S", "PVT->S", False, 0.55, -0.18),
           ("V", "cv", "VT->cv", False, -0.6, 0.0)]
X_NODES = {"X": (0.0, 1.2), "V": (1.6, 1.2), "D": (1.6, 0.0), "T": (3.4, 2.2), "H": (5.2, 2.2), "S": (5.2, 0.2),
           "cp": (3.4, 0.2)}
X_EDGES = [("X", "V", "TPX->V", True, 0.0, 0.14), ("V", "D", "V->D", False, -0.5, 0.0),
           ("V", "H", "TV->H", False, 0.0, 0.14), ("V", "cp", "TV->cp", False, -0.3, -0.26),
           ("V", "S", "TPV->S", False, 0.55, -0.18), ("S", "T", "PXS->T", True, 0.45, 0.0)]
R = 0.3


def _panel(ax, nodes, edges):
    fs.cartoon(ax)
    for name, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), R, fc=fs.CELL_FILL, ec=fs.BLACK, lw=0.8, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=fs.SMALL_PT, zorder=4)
    for a, b, key, newton, dx, dy in edges:
        (xa, ya), (xb, yb) = nodes[a], nodes[b]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                     color=fs.VERMILLION if newton else fs.GREEN, ls="--" if newton else "-",
                                     shrinkA=10, shrinkB=10, zorder=2))
        ax.text(0.5 * (xa + xb) + dx, 0.5 * (ya + yb) + dy, key, ha="center", va="center", fontsize=fs.SMALL_PT,
                family=fs.MONO, bbox=dict(boxstyle="round,pad=0.1", fc=fs.WHITE, ec="none"), zorder=5)
    ax.set_xlim(-0.5, 5.8)
    ax.set_ylim(-0.5, 2.7)


def make_fig():
    fig, axes = fs.new_figure("double", 2.9, 1, 2)
    _panel(axes[0], Y_NODES, Y_EDGES)
    _panel(axes[1], X_NODES, X_EDGES)
    fs.panel_labels(axes)
    axes[0].text(2.6, -0.45, "ThermoY (mass basis)", ha="center", va="center", fontsize=fs.SMALL_PT)
    axes[1].text(2.6, -0.45, "ThermoX (mole basis)", ha="center", va="center", fontsize=fs.SMALL_PT)
    return fig
