"""The scratch experiment behind "a function missing from the device table gives wrong values".

    python -P src/kintera_report/ch12/runs/func_table_gap.py --build <cuda build dir> --where end
    python -P src/kintera_report/ch12/runs/func_table_gap.py --build <cuda build dir> --where front

`^ch12-devfn-scratch` in `_device-functions.qmd` records a measurement made with `src/func_table.cpp`
"patched in a scratch copy only, not committed". That is why nobody could repeat it. This commits
the patch as something a script applies and then takes back, so the experiment is repeatable and the
tree is still clean afterwards.

What it does, in the footnote's own order:

  1. add a test function returning ln K = 7 and its derivative to `func1_table_cpu` and
     `func1_names` in `src/func_table.cpp` -- and to nothing else, so `func1_table_cuda` in
     `src/func_table.cu` keeps the old layout;
  2. `cmake --build . --target kintera_release`, so only the CPU library and its tables change;
  3. probe ln K on CPU and CUDA tensors;
  4. restore the source and rebuild, and check the restored library is byte-identical.

`--where end` appends, so the test function's device index `id + 1` points one past the end of the
device table. `--where front` inserts, so every later name shifts by two device slots.

The chapter's reading of the result -- that the slot past the end was null on that build -- is
labelled there as a hypothesis, and this script does not promote it: it reports what the two columns
say and leaves the interpretation where the chapter put it. Reading past the end of a device array
is undefined behaviour, so a different build or GPU may well print something else. That is worth
knowing and is the reason to keep the experiment runnable.

Needs a `-DCUDA=ON` build directory and a kintera checkout in `KINTERA_REPO`.
"""
import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

TEST_FN = "report_test_lnk7"
DECL = """
// ---- added by doc/tech-report/src/kintera_report/ch12/runs/func_table_gap.py; removed again ----
inline DISPATCH_MACRO double report_test_lnk7(double T) { return 7.0; }
inline DISPATCH_MACRO double report_test_lnk7_ddT(double T) { return 0.0; }
// ---- end of the scratch addition ----
"""


def patched_source(text: str, where: str) -> str:
    """add the test function to the CPU table and the name list, and to nothing else"""
    if TEST_FN in text:
        sys.exit("func_table_gap.py: the source already carries the scratch function; restore it first")
    out = text.replace("namespace kintera {", "namespace kintera {\n" + DECL, 1)
    if where == "end":
        out = out.replace("na_h2s_visscher, na_h2s_visscher_ddT};",
                          f"na_h2s_visscher, na_h2s_visscher_ddT,\n    {TEST_FN}, {TEST_FN}_ddT}};", 1)
        out = out.replace('"na_h2s_visscher",\n    "na_h2s_visscher_ddT"};',
                          f'"na_h2s_visscher",\n    "na_h2s_visscher_ddT", "{TEST_FN}", "{TEST_FN}_ddT"}};', 1)
    else:
        out = out.replace("user_func1 func1_table_cpu[] = {\n    h2o_ideal,",
                          f"user_func1 func1_table_cpu[] = {{\n    {TEST_FN}, {TEST_FN}_ddT,\n    h2o_ideal,", 1)
        out = out.replace('std::vector<std::string> func1_names = {\n    "h2o_ideal",',
                          'std::vector<std::string> func1_names = {\n'
                          f'    "{TEST_FN}", "{TEST_FN}_ddT",\n    "h2o_ideal",', 1)
    if TEST_FN not in out.split("func1_table_cpu")[1].split("}")[0]:
        sys.exit("func_table_gap.py: could not place the scratch function; the source has moved on "
                 "from what this script knows. Update it rather than guessing.")
    return out


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build(build_dir: pathlib.Path) -> None:
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "kintera_release"],
                   check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", required=True, help="a -DCUDA=ON Release build directory")
    ap.add_argument("--where", choices=["end", "front"], required=True)
    ap.add_argument("--names", default="h2o_ideal h2o_ideal_ddT nh3_ideal na_h2s_visscher_ddT")
    ap.add_argument("--dry-run", action="store_true", help="write the patched source and stop")
    args = ap.parse_args()

    repo = pathlib.Path(os.environ.get("KINTERA_REPO", ""))
    if not repo.exists():
        sys.exit("func_table_gap.py: set KINTERA_REPO to a kintera checkout")
    src = repo / "src" / "func_table.cpp"
    build_dir = pathlib.Path(args.build)
    if not args.dry_run and not (build_dir / "CMakeCache.txt").exists():
        sys.exit(f"func_table_gap.py: {build_dir} is not a cmake build directory")

    original = src.read_text()
    backup = src.with_suffix(".cpp.func_table_gap.bak")
    here = pathlib.Path(__file__).resolve().parent
    probe = here / "func_table_probe.py"

    shutil.copy2(src, backup)
    try:
        src.write_text(patched_source(original, args.where))
        print(f"patched {src} ({args.where}); CUDA table in src/func_table.cu untouched")
        if args.dry_run:
            return 0
        lib = next(build_dir.glob("lib/libkintera_release.so"), None)
        before = sha(lib) if lib else None
        build(build_dir)
        print("rebuilt kintera_release (CPU library and its tables only)")
        rc = subprocess.run([sys.executable, "-P", str(probe), *args.names.split()]).returncode
        print(f"probe exit {rc}")
    finally:
        shutil.move(str(backup), str(src))
        if not args.dry_run:
            build(build_dir)
            lib = next(build_dir.glob("lib/libkintera_release.so"), None)
            after = sha(lib) if lib else None
            same = (before is not None and before == after)
            print(f"restored {src} and rebuilt; library digest {before} -> {after}"
                  f" ({'bit-identical' if same else 'DIFFERS -- investigate before trusting the run'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
