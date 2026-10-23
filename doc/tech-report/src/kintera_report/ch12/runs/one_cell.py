"""RECOVERED, not newly written. Byte-for-byte the producer of the committed
derivation-checks/ch12/saved-preparation outputs, restored from the report's own history at
commit 516917515670447774cada7d52ba5cb705df4987 (removed by cfad2f6d36b1649b4253f48ca1aa780747ba2180).
Only this header is added; the code below is unchanged, so a rerun is comparable with the
saved records. Needs KINTERA_REPO pointing at a kintera checkout at the pin in code_pin.yml.
"""
"""Minimal deck-derived TP status disagreement; no implementation changes."""
from probe import *
m=module('test_cuda_workspace')
for tol in ['1.e-12','1.e-5']:
    path=OUT/'one-cell.yaml';path.write_text(m.CARD.replace('ftol: 1.e-12','ftol: '+tol))
    op=kt.ThermoOptions.from_yaml(str(path));outs=[]
    for dev in ['cpu','cuda']:
        th=kt.ThermoX(op);th.to(torch.device(dev));th.to(torch.float32)
        T=m.TEMP[1:2].to(dtype=torch.float32,device=dev)
        x=torch.zeros(1,17,dtype=torch.float32,device=dev);x[:,1:9]=.02;x[:,0]=1-x.sum(-1)
        d=torch.zeros(1,1,dtype=torch.float32,device=dev)
        g=th.forward(T,torch.full_like(T,1e5),x,False,d)
        outs.append((x.cpu(),g.cpu(),d.cpu()))
        emit(label='one_cell/status',ftol=tol,device=dev,T=float(T[0]),P=1e5,diag=float(d[0,0]),finite=bool(torch.isfinite(x).all()))
    cmp('one_cell/fractions/'+tol,outs[0][0],outs[1][0],torch.float32)
    cmp('one_cell/gain/'+tol,outs[0][1],outs[1][1],torch.float32)
