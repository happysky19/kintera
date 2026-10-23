"""K15 reach check: absent-gas exit gain of equilibrate_tp (issue #54).

Leg is chosen by $K15_LEG = pin | fix; $K15_WT points at the kintera checkout.
"""
import os
import sys

LEG = os.environ["K15_LEG"]
WT = os.environ["K15_WT"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), LEG, "_pyinstall"))

import torch  # noqa: E402
import kintera  # noqa: E402

torch.set_default_dtype(torch.float64)

DECKS = {
    "jupiter": "tests/jupiter.yaml",
    "earth": "tests/earth.yaml",
    "jup-thermo": "tests/jup-thermo.yaml",
    "kcl_mgsio3": "examples/jupiter_kcl_mgsio3.yaml",
}


def deck(name):
    return os.path.join(WT, DECKS[name])


def thermo(name, max_iter=None, ftol=None):
    op = kintera.ThermoOptions.from_yaml(deck(name))
    if max_iter is not None:
        op.max_iter(max_iter)
    if ftol is not None:
        op.ftol(ftol)
    th = kintera.ThermoX(op)
    th.to(torch.float64)
    return op, th


def names(op):
    """Working-set species names in xfrac order: vapours then clouds."""
    allnames = kintera.species_names()
    return [allnames[i] for i in list(op.vapor_ids()) + list(op.cloud_ids())]


def stoich(op):
    """Nucleation stoichiometry, species x reactions, in the working-set order."""
    return kintera.ThermoX(op).stoich.clone().to(torch.float64)


def logk_tilde(th, op, temp, pres, xfrac):
    """ln K~_r = Lambda_r - ln RH_r, measured with the pin's own svp functions."""
    conc = th.compute("TPX->V", [temp, pres, xfrac])
    lnrh = kintera.relative_humidity(temp, conc, th.stoich, op.nucleation(),
                                     len(list(op.vapor_ids())))
    S = th.stoich
    ngas = len(list(op.vapor_ids()))
    nuhat = -S[:ngas, :]                      # species x reaction, consumed > 0
    xg = xfrac[..., :ngas].sum(-1, keepdim=True)
    lnfrac = torch.log(xfrac[..., :ngas] / xg)
    lam = (lnfrac.unsqueeze(-1) * nuhat).sum(-2)
    return lam - torch.log(lnrh.clamp_min(1e-300)), lam


def equilibrate(th, temp, pres, x0):
    """Returns (gain, xout, diag). x0 is not modified."""
    x = x0.clone()
    diag = torch.zeros(x.shape[:-1])
    gain = th.forward(temp, pres, x, False, diag)
    return gain, x, diag


def rate_ddT_from_gain(gain, logsvp_ddT):
    """Exactly effective_cp's conversion: row-equilibrate, then gelsd lstsq."""
    row_scale = gain.abs().amax(-1, keepdim=True)
    row_scale = torch.where(row_scale > 0., row_scale, torch.ones_like(row_scale))
    gain_eq = gain / row_scale
    rhs_eq = logsvp_ddT / row_scale.squeeze(-1)
    return torch.linalg.lstsq(gain_eq, rhs_eq.unsqueeze(-1), driver="gelsd")[0].squeeze(-1)
