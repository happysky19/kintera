"""K42 matched float64 CPU/CUDA probe, run at the pin the report sets in `code_pin.yml`.
The supplied 2000-trial RNG sequence is unchanged. Floating-point U@V is
numerically rank deficient; this does not prove exact rank of stored entries.
No assertion of identical classifications or a universal singularity detector.
"""
import argparse,collections,hashlib,json,os,pathlib,platform,warnings
import numpy as np
import scipy,scipy.sparse as sp,scipy.sparse.linalg as spl
import torch,kintera as kt
from kintera.kintera import cuda_csr_solve_cusolver
# The expected pin is the report's own doc/tech-report/code_pin.yml, two directories up: the single
# source of truth, so this driver runs at whatever pin the report sets rather than at a literal.
PIN_FILE=pathlib.Path(__file__).resolve().parents[2]/'code_pin.yml'
def read_pin(path=PIN_FILE):
    for line in path.read_text().splitlines():
        if line.startswith('sha:'): return line.split(':',1)[1].strip()
    raise SystemExit(f'check_ch12_k42.py: no "sha:" line in {path}')
PIN=read_pin()
torch.set_num_threads(1)
def emit(x): print(json.dumps(x,allow_nan=False),flush=True)
def native(A,b):
    csr=sp.csr_matrix(A)
    cv=lambda x:torch.from_numpy(x.copy()).cuda()
    z=cuda_csr_solve_cusolver(cv(csr.indptr.astype(np.int32)),cv(csr.indices.astype(np.int32)),cv(csr.data),cv(b),0.,0)
    torch.cuda.synchronize();return z.cpu().numpy()
def measure(A,b,solve):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:x=solve()
        except RuntimeError as e:return {'status':'raised','error':str(e).splitlines()[0],'warnings':len(caught)}
    rec={'status':'finite_return' if np.isfinite(x).all() else 'nonfinite_return','warnings':len(caught),'nan':int(np.isnan(x).sum()),'inf':int(np.isinf(x).sum())}
    if rec['status']=='finite_return':
        aa=np.asarray(A,dtype=np.float64);xx=np.asarray(x,dtype=np.float64);bb=np.asarray(b,dtype=np.float64)
        res=aa@xx-bb;amax=float(np.linalg.norm(aa,np.inf));xmax=float(np.max(np.abs(xx)));bmax=float(np.max(np.abs(bb)));rmax=float(np.max(np.abs(res)))
        rec.update(max_abs_x=xmax,max_abs_residual=rmax,relative_residual_inf=rmax/bmax if bmax else rmax,backward_error_inf=rmax/(amax*xmax+bmax) if amax*xmax+bmax else 0.)
    return rec

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--records',type=pathlib.Path);args=parser.parse_args()
    import subprocess
    assert subprocess.check_output(['git','-C',os.environ['KINTERA_REPO'],'rev-parse','HEAD'],text=True).strip()==PIN
    emit({'pin':PIN,'python':platform.python_version(),'machine':platform.machine(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'seed':20261020,'dtype':'float64','trials':2000,'tol':0,'reorder':0,'threads':torch.get_num_threads()})
    rng=np.random.default_rng(20261020);records=[];digest=hashlib.sha256();dims=collections.Counter()
    for t in range(2000):
        n=int(rng.integers(3,7));r=n-1
        A=rng.standard_normal((n,r))@rng.standard_normal((r,n));b=rng.standard_normal(n)
        digest.update(np.asarray([n],dtype='<i8').tobytes());digest.update(A.astype('<f8').tobytes());digest.update(b.astype('<f8').tobytes());dims[n]+=1
        row={'trial':t,'n':n,'numerical_rank':int(np.linalg.matrix_rank(A)),'cond2':float(np.linalg.cond(A))}
        row['cpu']=measure(A,b,lambda:spl.factorized(sp.csc_matrix(A))(b));row['cuda']=measure(A,b,lambda:native(A,b));records.append(row)
    if args.records:args.records.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in records))
    emit({'input_sha256':digest.hexdigest(),'counts_by_dimension':dict(dims),'numerical_rank_n_minus_1':sum(r['numerical_rank']==r['n']-1 for r in records)})
    for dev in ['cpu','cuda']:
        stats={'backend':dev,'counts':dict(collections.Counter(r[dev]['status'] for r in records)),'warning_count':sum(r[dev]['warnings'] for r in records),'errors':dict(collections.Counter(r[dev].get('error') for r in records if r[dev]['status']=='raised'))}
        vals=[r[dev] for r in records if r[dev]['status']=='finite_return']
        for key in ['max_abs_x','max_abs_residual','relative_residual_inf','backward_error_inf']:
            a=np.array([v[key] for v in vals]);stats[key]={'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
        stats['finite_relative_residual_gt_1e_8']=sum(v['relative_residual_inf']>1e-8 for v in vals)
        stats['finite_backward_error_le_1e_12']=sum(v['backward_error_inf']<=1e-12 for v in vals)
        emit(stats)
    emit({'joint_classifications':dict(collections.Counter(r['cpu']['status']+'/'+r['cuda']['status'] for r in records))})
    for label,A in [('structural_diag',np.diag([4.,4.,0.])),('near_diag',np.diag([4.,4.,1e-14])),('exact_dependent_rows',np.array([[1.,2.,3.],[4.,5.,6.],[5.,7.,9.]]))]:
        b=np.ones(3);emit({'case':label,'A':A.tolist(),'b':b.tolist(),'cpu':measure(A,b,lambda:spl.factorized(sp.csc_matrix(A))(b)),'cuda':measure(A,b,lambda:native(A,b))})
    # Runtime library identities after cuSolver has been loaded; no home paths in integration output.
    libs=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(s in line for s in ['libcusolver.so','libcusparse.so','libcudart.so','libkintera','/kintera.cpython'])})
    emit({'loaded_library_sha256':{pathlib.Path(p).name:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in libs}})
    print('COMPLETE: measured classifications; no numerical pass/fail threshold asserted.',flush=True)
if __name__=='__main__':main()
