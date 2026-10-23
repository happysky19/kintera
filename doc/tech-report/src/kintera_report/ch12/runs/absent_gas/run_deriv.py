import torch, k15_common as K, deriv as D

op, th = K.thermo("jupiter", max_iter=D.MAXIT, ftol=D.FTOL)
sp = K.names(op); ngas = len(list(op.vapor_ids()))
def mk(**kw):
    x = torch.zeros(1, len(sp))
    for k,v in kw.items(): x[0, sp.index(k)] = v
    x[0,0] = 1.0 - x[0,1:].sum(); return x
xref = mk(H2O=2.5e-3, NH3=3e-4, H2S=8e-5)   # all gases present: for ln K only
HS = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]

CASES = [
  ("control all3   ", mk(H2O=2.5e-3, NH3=3e-4, H2S=8e-5), 200.0, 1e5),
  ("control all3   ", mk(H2O=2.5e-3, NH3=3e-4, H2S=8e-5), 160.0, 1e5),
  ("ABSENT noH2S   ", mk(H2O=2.5e-3, NH3=3e-4),           200.0, 1e5),
  ("ABSENT noH2S   ", mk(H2O=2.5e-3, NH3=3e-4),           160.0, 1e5),
  ("ABSENT noH2S   ", mk(H2O=2.5e-3, NH3=3e-4),           130.0, 1e4),
  ("ABSENT noNH3   ", mk(H2O=2.5e-3, H2S=8e-5),           180.0, 1e5),
  ("ABSENT noH2O   ", mk(NH3=3e-4, H2S=8e-5),             160.0, 1e5),
  ("ABSENT nN_nS   ", mk(H2O=2.5e-3),                     200.0, 1e5),
]
for label, x0, T, p in CASES:
    g, xe, d, rhs, pred, fds = D.compare(th, op, T, p, x0, xref, HS)
    zero = [sp[i] for i in range(ngas) if xe[0,i].item()==0.0]
    cl = {sp[i]: xe[0,i].item() for i in range(ngas, len(sp)) if xe[0,i].item()>0}
    print(f"\n== {label} T={T} p={p:.0e} diag={d:.0f} zero_gases={zero} clouds={ {k:f'{v:.3e}' for k,v in cl.items()} }")
    print(f"   rhs dlnK/dT = {[f'{v:.6e}' for v in rhs.tolist()]}")
    print(f"   gain rows   = {[[f'{v:.4e}' for v in row] for row in g.tolist()]}")
    print(f"   pred dxi/dT = {[f'{v:.9e}' for v in pred.tolist()]}")
    for h, fd, diags, resid in fds:
        rel = [(abs(fd[r].item()-pred[r].item())/max(abs(pred[r].item()),abs(fd[r].item()),1e-300)) if max(abs(fd[r].item()),abs(pred[r].item()))>0 else 0.0 for r in range(len(pred))]
        print(f"   h={h:5.2f} fd={[f'{v:.9e}' for v in fd.tolist()]} relerr={[f'{v:.2e}' for v in rel]} diags={diags} lsqresid={resid:.1e}")
