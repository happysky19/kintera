import os, torch, sweep as S
Ts = [110,130,150,170,190,210,230,250,270,290,310,350,400,600,900,1200,1600]
ps = [1e2,1e3,1e4,1e5,1e6,1e7]
AM = [1e-5, 8e-5, 3e-4, 2.5e-3, 3e-2]
print("LEG", os.environ["K15_LEG"], "WT", os.environ["K15_WT"])
tot=dict(cells=0,absent=0,conv=0,vc=0,vg=0)
for name in ["jupiter","earth","jup-thermo","kcl_mgsio3"]:
    try:
        r = S.structural_scan(name, Ts, ps, AM)
    except Exception as e:
        print(f"{name}: SKIP ({type(e).__name__}: {e})"); continue
    print(f"\n{name}: nrxn={r['nrxn']} ngas={r['ngas']} species={r['species']}")
    print(f"  cells probed {r['cells']}, absent-gas reaction instances {r['absent']}, "
          f"of which at a converged exit {r['conv_absent']}")
    print(f"  VIOLATIONS cloud_present_with_absent_gas={len(r['viol_cloud'])}  "
          f"nonzero_gain_row_or_col={len(r['viol_gain'])}")
    for v in r['viol_cloud'][:5]: print("    cloud:", v)
    for v in r['viol_gain'][:5]:  print("    gain :", v)
    tot['cells']+=r['cells']; tot['absent']+=r['absent']; tot['conv']+=r['conv_absent']
    tot['vc']+=len(r['viol_cloud']); tot['vg']+=len(r['viol_gain'])
print("\nTOTAL", tot)
