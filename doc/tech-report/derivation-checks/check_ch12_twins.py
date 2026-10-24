"""Ch. 12 CPU/CUDA twins at the code pin, checked without a GPU.

Static part: parse the .cpp/.cu twins and compare what must match (function
tables, dispatch registrations, per-cell argument unpacking, sentinel name swap,
workspace size functions against the pmalloc sequences of the per-cell solvers).
Numeric part: run the per-cell implicit step (the same template the CUDA kernel
runs, here through the CPU driver) against a numpy solve of
(I/dt - S J) delta = S r, and its singular fallback against the numpy
minimum-norm least-squares solution. Also recompute the report's block-size
arithmetic from the code's formula.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import version_line, kintera as kt, torch, WT  # noqa: E402

print(version_line())


def src(rel):
    with open(os.path.join(WT, rel)) as f:
        return f.read()


def ok(flag):
    return "PASS" if flag else "FAIL"


def brace_list(text, name):
    m = re.search(name + r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
    return [t.strip().strip('"') for t in m.group(1).split(",") if t.strip()]


# (a) func1 tables: names == host table; device table == [nullptr] + names
cpp, cu = src("src/func_table.cpp"), src("src/func_table.cu")
names = [t.strip().strip('"') for t in re.search(
    r"func1_names\s*=\s*\{(.*?)\};", cpp, re.S).group(1).split(",") if t.strip()]
host = brace_list(cpp, "func1_table_cpu")
dev = brace_list(cu, "func1_table_cuda")
print(f"(a) func1: {len(names)} names, {len(host)} host, {len(dev)} device slots; "
      f"host==names {host == names}, device==[nullptr]+names {dev == ['nullptr'] + names}  "
      f"{ok(host == names and dev == ['nullptr'] + names)}")

# (b) func2/func3: device table == [nullptr] + host table, the same relation (a) checks for func1.
# Both tables were empty up to pin 0d9f67f9, where [nullptr] + [] is exactly the placeholder-only
# device table that used to be asserted; since e357792d func2 carries the H2 rotational entries, so
# emptiness is no longer the criterion while the index relation still is.
h2, h3 = brace_list(cpp, "func2_table_cpu"), brace_list(cpp, "func3_table_cpu")
d2, d3 = brace_list(cu, "func2_table_cuda"), brace_list(cu, "func3_table_cuda")
print(f"(b) func2/func3: host {h2}/{h3}, device {d2}/{d3}  "
      f"{ok(d2 == ['nullptr'] + h2 and d3 == ['nullptr'] + h3)}")

# (c) every stub defined in a .cpp has a CPU registration and a CUDA registration
pairs = ["src/kinetics/evolve_implicit_dispatch", "src/thermo/thermo_dispatch",
         "src/equilibrium/equilibrium_dispatch", "src/utils/utils_dispatch"]
bad = []
nstub = 0
for p in pairs:
    c, g = src(p + ".cpp"), src(p + ".cu")
    for stub in re.findall(r"^DEFINE_DISPATCH\((\w+)\)", c, re.M):
        nstub += 1
        if not re.search(r"REGISTER_ALL_CPU_DISPATCH\(\s*" + stub + r"\s*,", c):
            bad.append(stub + ":cpu")
        if not re.search(r"^REGISTER_CUDA_DISPATCH\(\s*" + stub + r"\s*,", g, re.M):
            bad.append(stub + ":cuda")
print(f"(c) {nstub} stubs, each with a CPU and a CUDA registration; missing {bad}  {ok(nstub > 0 and not bad)}")


# (d) per-cell unpacking: same (name, data index) sequence in each CPU/CUDA twin
def unpack(body):
    return re.findall(r"auto\s+(\w+)\s*=\s*\n?\s*reinterpret_cast<[^>]+>\(data\[(\d)\]", body)


def fn_body(text, fname):
    i = text.index("void " + fname + "(")
    j = text.find("\nvoid ", i + 5)
    return text[i:j if j > 0 else len(text)]


twins = [("src/kinetics/evolve_implicit_dispatch", "call_evolve_implicit"),
         ("src/thermo/thermo_dispatch", "call_equilibrate_tp"),
         ("src/thermo/thermo_dispatch", "call_equilibrate_uv"),
         ("src/thermo/thermo_dispatch", "call_logsvp_inline"),
         ("src/equilibrium/equilibrium_dispatch", "call_equilibrium"),
         ("src/utils/utils_dispatch", "call_func1"),
         ("src/utils/utils_dispatch", "call_func2"),
         ("src/utils/utils_dispatch", "call_func3")]
mism = []
for p, f in twins:
    a = unpack(fn_body(src(p + ".cpp"), f + "_cpu"))
    b = unpack(fn_body(src(p + ".cu"), f + "_cuda"))
    if a != b or not a:
        mism.append((f, a, b))
print(f"(d) {len(twins)} driver twins unpack the same pointers in the same data[] order; mismatches {mism}  {ok(not mism)}")

# (e) sentinel swap of inline 'ideal'/'antoine' names present in both TP/UV twins
sw = 'if (s == "ideal" || s == "antoine") s = "h2o_ideal";'
td_c, td_g = src("src/thermo/thermo_dispatch.cpp"), src("src/thermo/thermo_dispatch.cu")
print(f"(e) sentinel name swap: cpp {td_c.count(sw)}, cu {td_g.count(sw)} (expected 2 and 2)  "
      f"{ok(td_c.count(sw) == 2 and td_g.count(sw) == 2)}")


# (f) workspace: size functions list the same allocations, in order, as the per-cell pmalloc calls
def norm(s):
    return re.sub(r"\s+", "", s)


def pm_sizes(text, start, stop):
    seg = text[text.index(start):text.index(stop, text.index(start))]
    return [norm(x) for x in re.findall(r"pmalloc(?:<Backend>)?\(work,\s*([^;]*?)\);", seg)]


def bump_sizes(text, fname):
    i = text.index(fname)
    seg = text[i:text.index("return", i)]
    return [norm(x) for x in re.findall(r"bump\(alignof\(\w+\),\s*([^;]*?)\);", seg)]


al = src("src/utils/alloc.h")
ei = src("src/kinetics/evolve_implicit_impl.h")
tp = src("src/thermo/equilibrate_tp.h")
uv = src("src/thermo/equilibrate_uv.h")
kk = src("src/math/leastsq_kkt.h")
ps = src("src/math/psolve.h")
ph = src("src/equilibrium/phase_equilibrate_tp.h")
eq = src("src/equilibrium/equilibrium_dispatch.cu")
cases = [
    ("evolve_implicit", pm_sizes(ei, "evolve_implicit_cell(", "mmdot"), bump_sizes(ei, "size_t evolve_implicit_space")),
    ("psolve", pm_sizes(ps, "void psolve(", "pool_rewind"), bump_sizes(al, "size_t psolve_space")),
    ("equilibrate_tp", pm_sizes(tp, "size_t mark", "pool_rewind"), bump_sizes(al, "size_t equilibrate_tp_space")),
    ("equilibrate_uv", pm_sizes(uv, "intEng = (T*)pmalloc", "pool_rewind"), bump_sizes(al, "size_t equilibrate_uv_space")),
    ("leastsq_kkt", pm_sizes(kk, "aug = (T*)pmalloc", "at_lu = "), bump_sizes(al, "size_t leastsq_kkt_space")),
    ("kkt_feasible_origin", pm_sizes(kk, "at_lu = ", "pool_rewind"), bump_sizes(al, "size_t leastsq_kkt_feasible_origin_space")),
    ("phase_equilibrate_tp", pm_sizes(ph, "size_t mark", "pool_rewind"), None),
]
# variable renames between the two sides (size = n2 + n3 in leastsq_kkt)
for name, pm, bp in cases:
    if name == "phase_equilibrate_tp":
        i = eq.index("size_t equilibrium_space")
        bp = [norm(x) for x in re.findall(r"bump\(alignof\(T\),\s*([^;]*?)\);", eq[i:eq.index("return", i)])]
    same = pm == bp
    print(f"(f-{name}) {len(pm)} pmalloc calls vs {len(bp)} size-function terms, same sizes in order: {same}  {ok(same)}")
uvp = pm_sizes(uv, "equilibrate_uv_partition(", "pool_rewind")
print(f"(f-uv_partition) pmalloc sizes {uvp}; size function 2*ns + nr  "
      f"{ok(uvp == ['nspecies*sizeof(T)', 'nspecies*sizeof(T)', 'nreaction*sizeof(T)'])}")

# (g) bytes per thread of the implicit step = 32 ns^2 + 48 ns (double), report eq-ch12-kernels-implicit-bytes
def implicit_bytes(n, sz=8):
    r8 = lambda b: -(-max(b, 8) // 8) * 8  # noqa: E731
    return 2 * r8(n * n * sz) + 2 * r8(n * sz) + 2 * r8(n * n * sz) + 4 * r8(n * sz)


okg = all(implicit_bytes(n) == 32 * n * n + 48 * n for n in range(1, 60))
print(f"(g) implicit-step scratch from the size function equals 32 ns^2 + 48 ns for ns = 1..59  {ok(okg)}")

# (h) block size N = min(32, max(1, M // b)), launch error iff b > M; report: M = 49152
M = 49152
thr = {n: min(32, max(1, M // implicit_bytes(n))) for n in (6, 7, 38, 39)}
err = {n: thr[n] * implicit_bytes(n) > M for n in (38, 39)}
okh = thr[6] == 32 and thr[7] < 32 and not err[38] and err[39]
print(f"(h) M=49152: threads ns=6 {thr[6]}, ns=7 {thr[7]}; launch error ns=38 {err[38]}, ns=39 {err[39]} "
      f"(report: 32 up to 6, error from 39)  {ok(okh)}")

# (i) per-cell implicit step (CPU driver, HostBump pool) against numpy, regular cells
rng = np.random.default_rng(12)
ns, nr, ncell, dt = 5, 4, 64, 0.3
S = rng.integers(-2, 3, size=(ns, nr)).astype(float)
J = rng.normal(size=(ncell, nr, ns))
r = rng.normal(size=(ncell, nr))
d = kt.evolve_implicit(torch.tensor(r), torch.tensor(S), torch.tensor(J), dt).numpy()
ref = np.stack([np.linalg.solve(np.eye(ns) / dt - S @ J[k], S @ r[k]) for k in range(ncell)])
e = np.max(np.abs(d - ref)) / np.max(np.abs(ref))
print(f"(i) evolve_implicit vs numpy solve, {ncell} cells ns={ns} nr={nr}: max rel diff {e:.2e}  tol 1e-12  {ok(e < 1e-12)}")

# (j) singular cell: A = I/dt - S J singular -> psolve fallback = minimum-norm least squares
dt = 1.0
S1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # ns=3, nr=2
J1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])  # S J = diag(1, .5, 0): A = diag(0, .5, 1)
r1 = np.array([0.7, -0.4])
d1 = kt.evolve_implicit(torch.tensor(r1)[None], torch.tensor(S1), torch.tensor(J1)[None], dt).numpy()[0]
A1 = np.eye(3) / dt - S1 @ J1
ref1 = np.linalg.pinv(A1) @ (S1 @ r1)
e1 = np.max(np.abs(d1 - ref1))
print(f"(j) singular cell, fallback vs numpy pinv: got {np.round(d1, 12)}, ref {np.round(ref1, 12)}, "
      f"max abs diff {e1:.1e}  tol 1e-12  {ok(e1 < 1e-12)}")

# (k) float32 path of the same per-cell function, against the float64 numpy result
d32 = kt.evolve_implicit(torch.tensor(r, dtype=torch.float32), torch.tensor(S, dtype=torch.float32),
                         torch.tensor(J, dtype=torch.float32), 0.3).double().numpy()
e32 = np.max(np.abs(d32 - ref)) / np.max(np.abs(ref))
print(f"(k) float32 evolve_implicit vs float64 numpy: max rel diff {e32:.2e} (INFO: sets the float32 CPU-vs-CUDA "
      f"comparison scale, not a CUDA result)  INFO")
