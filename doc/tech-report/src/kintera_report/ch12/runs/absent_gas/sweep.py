"""K15 full sweep: all in-tree nucleation decks, absent-gas reach + derivative check."""
import itertools, torch, k15_common as K, deriv as D

TOL_STRUCT = 0


def deck_info(name):
    op, th = K.thermo(name, max_iter=D.MAXIT, ftol=D.FTOL)
    sp = K.names(op); ngas = len(list(op.vapor_ids())); S = th.stoich
    rg = {r: [i for i in range(ngas) if S[i, r] != 0] for r in range(S.shape[1])}
    cl = {r: [i for i in range(ngas, S.shape[0]) if S[i, r] > 0] for r in range(S.shape[1])}
    gp = {r: [i for i in range(ngas) if S[i, r] > 0] for r in range(S.shape[1])}
    return op, th, sp, ngas, S, rg, cl, gp


def base_x(sp, ngas, amount):
    x = torch.zeros(1, len(sp))
    for i in range(1, ngas):
        x[0, i] = amount
    x[0, 0] = 1.0 - x[0, 1:].sum()
    return x


def variants(sp, ngas, amount):
    """Full-vapour state plus every subset with one or two vapours set to zero."""
    out = {"all": base_x(sp, ngas, amount)}
    idx = list(range(1, ngas))
    for k in (1, 2):
        for drop in itertools.combinations(idx, k):
            x = base_x(sp, ngas, amount)
            for i in drop:
                x[0, i] = 0.0
            x[0, 0] = 1.0 - x[0, 1:].sum()
            out["no_" + "_".join(sp[i] for i in drop)] = x
    return out


def structural_scan(name, Ts, ps, amounts):
    op, th, sp, ngas, S, rg, cl, gp = deck_info(name)
    n_cell = 0; n_absent = 0; n_conv_absent = 0; n_frozen = 0
    viol_cloud = []; viol_gain = []
    for amount in amounts:
        vs = variants(sp, ngas, amount)
        for lab, x0 in vs.items():
            for T, p in itertools.product(Ts, ps):
                n_cell += 1
                Tt = torch.tensor([float(T)]); pt = torch.tensor([float(p)])
                g, xe, d = D.equil_x(th, Tt, pt, x0)
                conv = d >= 1
                for r, gases in rg.items():
                    zero = [i for i in gases if xe[0, i].item() == 0.0]
                    if not zero:
                        continue
                    n_absent += 1
                    if not conv:
                        continue
                    n_conv_absent += 1
                    cloud_amt = max(xe[0, i].item() for i in cl[r]) if cl[r] else 0.0
                    absent_prod_gas = any(xe[0, i].item() == 0.0 for i in gp[r])
                    if absent_prod_gas:
                        n_frozen += 1
                    if cloud_amt > TOL_STRUCT and not absent_prod_gas:
                        viol_cloud.append((lab, T, p, r, cloud_amt, d,
                                           [sp[i] for i in zero]))
                    gz = g[r].abs().max().item() + g[:, r].abs().max().item()
                    if gz > 0.0:
                        viol_gain.append((lab, T, p, r, gz, d))
    return dict(deck=name, species=sp, ngas=ngas, nrxn=S.shape[1], cells=n_cell,
                absent=n_absent, conv_absent=n_conv_absent, frozen=n_frozen,
                viol_cloud=viol_cloud, viol_gain=viol_gain)
