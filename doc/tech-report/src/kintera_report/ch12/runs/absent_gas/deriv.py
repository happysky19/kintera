"""K15: convert the returned gain to d xi/dT and compare with finite differences.

Units: the returned gain acts on extents measured as fractions of the total at the
final (renormalised) state, so the linearised composition map there is
    M[i,r] = S[i,r] - x_out[i] * sum_i S[i,r]
and the finite-difference derivative is  dxi/dT = lstsq(M, dx_out/dT).
The gain prediction is exactly effective_cp's conversion (extrapolate_ad.cpp:41-57).
"""
import torch, k15_common as K

MAXIT, FTOL = 400, 1e-13


def lnK(th, op, T, p, xref):
    """ln K_r(T), composition independent: Lambda - ln RH at a fully present state."""
    lk, _ = K.logk_tilde(th, op, T, p, xref)
    S = th.stoich
    ngas = len(list(op.vapor_ids()))
    nusum = (-S[:ngas, :]).sum(0)
    return lk + nusum * torch.log(p).unsqueeze(-1)


def logsvp_ddT(th, op, T, p, xref, h):
    a = lnK(th, op, T + h, p, xref)
    b = lnK(th, op, T - h, p, xref)
    return (a - b) / (2 * h)


def equil_x(th, T, p, x0):
    g, xo, d = K.equilibrate(th, T, p, x0)
    n = th.stoich.shape[1]
    return g.reshape(n, n), xo, d.item()


def fd_rate(th, T, p, x0, h, mode="central"):
    """dxi/dT by finite difference of the equilibrium composition."""
    if mode == "central":
        _, xp, dp = equil_x(th, T + h, p, x0)
        _, xm, dm = equil_x(th, T - h, p, x0)
        _, x0e, d0 = equil_x(th, T, p, x0)
        dx = (xp - xm) / (2 * h); diags = (dm, d0, dp)
    elif mode == "forward":
        _, xp, dp = equil_x(th, T + h, p, x0)
        _, x0e, d0 = equil_x(th, T, p, x0)
        dx = (xp - x0e) / h; diags = (d0, d0, dp)
    else:
        _, xm, dm = equil_x(th, T - h, p, x0)
        _, x0e, d0 = equil_x(th, T, p, x0)
        dx = (x0e - xm) / h; diags = (dm, d0, d0)
    S = th.stoich
    M = S.unsqueeze(0) - x0e.unsqueeze(-1) * S.sum(0).reshape(1, 1, -1)
    rate = torch.linalg.lstsq(M, dx.unsqueeze(-1), driver="gelsd")[0].squeeze(-1)
    resid = (M.matmul(rate.unsqueeze(-1)).squeeze(-1) - dx).abs().max().item()
    return rate, x0e, diags, resid


def compare(th, op, T, p, x0, xref, hs, mode="central", hK=1e-3):
    Tt = torch.tensor([float(T)]); pt = torch.tensor([float(p)])
    g, xe, d = equil_x(th, Tt, pt, x0)
    rhs = logsvp_ddT(th, op, Tt, pt, xref, hK)
    pred = K.rate_ddT_from_gain(g.unsqueeze(0), rhs)[0]
    out = []
    for h in hs:
        fd, xe2, diags, resid = fd_rate(th, Tt, pt, x0, h, mode)
        out.append((h, fd[0], diags, resid))
    return g, xe, d, rhs[0], pred, out
