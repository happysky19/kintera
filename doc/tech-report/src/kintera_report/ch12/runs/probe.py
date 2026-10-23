"""RECOVERED, not newly written. Byte-for-byte the producer of the committed
derivation-checks/ch12/saved-preparation outputs, restored from the report's own history at
commit 516917515670447774cada7d52ba5cb705df4987 (removed by cfad2f6d36b1649b4253f48ca1aa780747ba2180).
Only this header is added; the code below is unchanged, so a rerun is comparable with the
saved records. Needs KINTERA_REPO pointing at a kintera checkout at the pin in code_pin.yml.
"""
"""Bounded matched-input CPU/CUDA probes. JSONL preserves every comparison.
Not a physical-model validation. Run separately per group via run.sh.
"""
import argparse, importlib.util, json, os, pathlib, sys, traceback
import torch
import kintera as kt
ROOT=pathlib.Path(os.environ['KINTERA_REPO']); OUT=pathlib.Path(__file__).resolve().parent
SEED=20261020
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
def emit(**kw): print(json.dumps(kw,allow_nan=False),flush=True)
def module(name):
    s=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def cmp(label,a,b,dtype,**extra):
    a=a.detach().cpu();b=b.detach().cpu();fin=torch.isfinite(a)&torch.isfinite(b)
    delta=(a.double()-b.double()).abs(); df=delta[fin]
    atol,rtol=(1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-4)
    den=torch.maximum(a.double().abs(),b.double().abs()).clamp_min(atol)
    emit(label=label,dtype=str(dtype),count=a.numel(),finite_cpu=int(torch.isfinite(a).sum()),finite_cuda=int(torch.isfinite(b).sum()),finite_mask_mismatch=int((torch.isfinite(a)!=torch.isfinite(b)).sum()),nan_mask_mismatch=int((torch.isnan(a)!=torch.isnan(b)).sum()),inf_sign_mismatch=int(((torch.isinf(a)|torch.isinf(b))&(a!=b)).sum()),max_abs=float(df.max()) if df.numel() else None,max_rel_symmetric=float((delta/den)[fin].max()) if df.numel() else None,max_abs_over_cpu_max=float(df.max()/a[fin].double().abs().max().clamp_min(atol)) if df.numel() else None,outside_tolerance=int((fin&(delta>atol+rtol*a.double().abs())).sum()),atol=atol,rtol=rtol,**extra)

def kinetics(dtype):
    for name in ['test_evaporation_kessler','test_coagulation_pressure_switch','jupiter']:
        if name=='jupiter': path=ROOT/'tests/jupiter.yaml'
        else:
            m=module(name);path=OUT/(name+'.yaml');path.write_text(m.CARD.format(extra=m.KESSLER) if 'kessler' in name else m.CARD.format(**m.SWITCH))
        op=kt.KineticsOptions.from_yaml(str(path));n=len(op.species());gen=torch.Generator().manual_seed(SEED)
        T=(80+40*torch.rand(64,generator=gen)).to(dtype) if name!='jupiter' else (180+40*torch.rand(64,generator=gen)).to(dtype)
        P=(10**(3.5+torch.rand(64,generator=gen))).to(dtype)
        p0=torch.tensor(1e4,dtype=dtype);P[:3]=torch.stack([torch.nextafter(p0,torch.tensor(0.,dtype=dtype)),p0,torch.nextafter(p0,torch.tensor(float('inf'),dtype=dtype))])
        C=(1e-3*torch.rand(64,n,generator=gen)).to(dtype)
        if 'kessler' in name:
            # Include both sides of the evaporation activation at identical T/C/P.
            names=op.species(); sat=torch.tensor([m.svp(float(t))/(m.R*float(t)) for t in T],dtype=dtype)
            C[:,names.index('CH4')]=sat*torch.linspace(.5,1.5,64,dtype=dtype)
        out=[]
        for dev in ['cpu','cuda']:
            kin=kt.Kinetics(op);kin.to(torch.device(dev));kin.to(dtype)
            t,p,c=[v.to(dev) for v in (T,P,C)]
            r,dc,dt,k=kin.forward_with_rate_constants(t,p,c)
            j=kin.jacobian(t,c,torch.ones_like(t),r,dc,dt)
            step=kt.evolve_implicit(r,kin.stoich,j,10.)
            out.append([v.detach().cpu() if v is not None else torch.empty(0,dtype=dtype) for v in [r,dc,dt,k,j,step]])
        for label,a,b in zip(['rate','dkdC','dkdT','kf','jacobian','step'],*out):cmp(name+'/'+label,a,b,dtype,seed=SEED,cells=64)
        emit(label=name+'/rate_branch',dtype=str(dtype),zero_mask_mismatch=int(((out[0][0]==0)!=(out[1][0]==0)).sum()),positive_cpu=int((out[0][0]>0).sum()),positive_cuda=int((out[1][0]>0).sum()))
        if name!='jupiter':
            emit(label=name+'/conservation',dtype=str(dtype),cpu_max_abs_sum_delta=float(out[0][-1].sum(-1).abs().max()),cuda_max_abs_sum_delta=float(out[1][-1].sum(-1).abs().max()),definition='sum of species delta c; every reaction is 1:1, hence sum must vanish')

def workspace(dtype):
    m=module('test_cuda_workspace');path=OUT/'workspace.yaml';path.write_text(m.CARD.replace('ftol: 1.e-12', 'ftol: '+os.environ.get('WORKSPACE_FTOL','1.e-12')))
    T=m.TEMP.to(dtype); n=T.numel()
    for mode in ['TP','UV-partition','UV-kkt']:
        op=kt.ThermoOptions.from_yaml(str(path))
        if mode!='TP':op.uv_solver(mode.split('-')[1])
        # Construct UV energy once on CPU; do not recompute inputs on CUDA.
        if mode!='TP':
            th=kt.ThermoY(op);th.to(dtype);rho=torch.ones(n,dtype=dtype);y=torch.zeros(16,n,dtype=dtype);y[:8]=.02
            E=th.compute('VT->U',[th.compute('DY->V',[rho,y]),T])
        out=[]
        for dev in ['cpu','cuda']:
            th=(kt.ThermoX if mode=='TP' else kt.ThermoY)(op);th.to(torch.device(dev));th.to(dtype)
            diag=torch.zeros(n,1,device=dev,dtype=dtype)
            if mode=='TP':
                x=torch.zeros(n,17,dtype=dtype,device=dev);x[:,1:9]=.02;x[:,0]=1-x.sum(-1);before=x.clone()
                gain=th.forward(T.to(dev),torch.full_like(T,1e5).to(dev),x,False,diag)
                cons=(x[:,1:9]+x[:,9:]-before[:,1:9]-before[:,9:]).abs().max();cloud=x[:,9:]>0
                extras={}
            else:
                x=y.to(dev).clone();before=x.clone();gain=th.forward(rho.to(dev),E.to(dev),x,False,diag)
                cons=(x[:8]+x[8:]-before[:8]-before[8:]).abs().max();cloud=x[8:]>0
                extras={}
            out.append((x.cpu(),gain.cpu(),diag.cpu(),cloud.cpu()))
            emit(label=mode+'/status',dtype=str(dtype),device=dev,cells=n,diag_values={str(float(v)):int((diag==v).sum()) for v in diag.unique()},max_pair_conservation_error=float(cons),cloudy=int(cloud.sum()),**extras)
        for i,label in enumerate(['fractions','gain']):cmp(mode+'/'+label,out[0][i],out[1][i],dtype,cells=n)
        emit(label=mode+'/branches',dtype=str(dtype),diag_exact_mismatch=int((out[0][2]!=out[1][2]).sum()),success_mask_mismatch=int(((out[0][2]>=0)!=(out[1][2]>=0)).sum()),cloud_mask_mismatch=int((out[0][3]!=out[1][3]).sum()))
        both=(out[0][2].flatten()>=0)&(out[1][2].flatten()>=0)
        for mask,label in [(both,'both_success'),(~both,'either_failed')]:
            cmp(mode+'/gain/'+label,out[0][1][mask],out[1][1][mask],dtype,cells=int(mask.sum()))
        mism=(out[0][2]!=out[1][2]).flatten()
        emit(label=mode+'/mismatch_cells',dtype=str(dtype),ftol=os.environ.get('WORKSPACE_FTOL','1.e-12'),cells=[{'index':i,'T':float(T[i]),'cpu_diag':float(out[0][2][i,0]),'cuda_diag':float(out[1][2][i,0])} for i in torch.where(mism)[0].tolist()])


def functions(dtype):
    # Native value AND named derivative entries; log(rh) recovers the dispatched function.
    # These are arithmetic stress domains, not claims of empirical fit validity.
    names=['h2o_ideal','h2o_bryan','nh3_ideal','nh3_h2s_lewis','h2s_ideal','h2s_antoine','ch4_ideal','so2_antoine','co2_antoine','kcl_lodders','na_h2s_visscher']
    for name in names:
        base=torch.linspace(100.,1500.,31,dtype=dtype)
        boundaries={'h2o_ideal':273.16,'nh3_ideal':195.4,'h2s_ideal':187.63,'h2s_antoine':212.8,'ch4_ideal':90.67}
        if name in boundaries:
            x=torch.tensor(boundaries[name],dtype=dtype);base=torch.cat([base,torch.stack([torch.nextafter(x,torch.zeros_like(x)),x,torch.nextafter(x,torch.full_like(x,float('inf')))])])
        for suffix in ['', '_ddT']:
            op=kt.NucleationOptions().logsvp([name+suffix]);out=[]
            for dev in ['cpu','cuda']:
                t=base.to(dev);rh=kt.relative_humidity(t,torch.ones(len(t),2,dtype=dtype,device=dev),torch.zeros(2,1,dtype=dtype,device=dev),op,1)
                out.append(-rh.log())
            cmp(name+suffix,out[0],out[1],dtype,temperature_min=float(base.min()),temperature_max=float(base.max()))

def sparse(dtype):
    gen=torch.Generator().manual_seed(SEED)
    for label,n in [('in_tree_12',12),('tridiagonal_63',63),('singular_3',3),('near_singular_3',3)]:
        A=4*torch.eye(n,dtype=dtype)
        if n==12:A[0,1]=-.1;A[5,3]=.2
        elif n==63:
            A+=torch.diag(torch.full((n-1,),-1.,dtype=dtype),1)+torch.diag(torch.full((n-1,),-1.,dtype=dtype),-1)
        else:A[-1,-1]=0 if label=='singular_3' else 1e-14
        b=torch.randn(1,n,1,generator=gen,dtype=torch.float64).to(dtype);out=[]
        for dev in ['cpu','cuda']:
            try:
                mat=kt.SparseSystemMatrix.from_dense(A.to(dev),ncol=1,nlyr=n,nspecies=1)
                sol=kt.solve_sparse_system(mat,b.to(dev)).cpu();out.append(sol)
                emit(label=label+'/status',dtype=str(dtype),device=dev,status='returned',finite=int(torch.isfinite(sol).sum()),max_residual=float((A.double()@sol.double().flatten()-b.double().flatten()).abs().max()))
            except Exception as e:out.append(None);emit(label=label+'/status',dtype=str(dtype),device=dev,status='exception',error=str(e).splitlines()[0])
        if all(v is not None for v in out):cmp(label,*out,dtype)
        else:emit(label=label+'/status_agreement',dtype=str(dtype),same_return_status=(out[0] is None)==(out[1] is None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['kinetics','workspace','functions','sparse']);p.add_argument('--dtype',default='float64',choices=['float32','float64']);a=p.parse_args()
    globals()[a.group](getattr(torch,a.dtype))
    torch.cuda.synchronize()
