"""The scratch experiment behind "a function missing from the device table gives wrong values".

    python -P func_table_gap.py --build <cuda build dir> --where end
    python -P func_table_gap.py --build <cuda build dir> --where front

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


def cache_value(build_dir: pathlib.Path, key: str) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(key + ":"):
            return line.split("=", 1)[1].strip()
    return None


def fresh_configure(build_dir: pathlib.Path, source: pathlib.Path, reuse: bool) -> None:
    """Configure this build directory from scratch, for this source tree.

    A disposable checkout is normally made with `cp -a`, which brings the build directory along.
    Everything in it -- the cache, the generated makefiles, and a CMakeCache inside every
    FetchContent `_deps/*-subbuild` -- still names the ORIGINAL tree. `cmake --build` then prints
    "Build files have been written to: <original>/build-cuda", rebuilds nothing here, and exits 0,
    so the whole cycle reports success while the library is never rebuilt. Repairing that
    piecemeal is not worth it: the only state that is certainly ours is state we generated, so by
    default the directory is removed and configured again. `--reuse-build` skips this for a build
    directory you know was made here.
    """
    if reuse:
        print("  --reuse-build: configuring skipped; the build directory must already be this tree's")
        return
    if build_dir.exists():
        shutil.rmtree(build_dir)
    args = ["cmake", "-S", str(source.resolve()), "-B", str(build_dir.resolve()),
            "-DCMAKE_BUILD_TYPE=Release", "-DCUDA=ON", "-DNETCDF=ON"]
    for env, flag in (("CMAKE_CUDA_ARCHITECTURES", "-DCMAKE_CUDA_ARCHITECTURES="),
                      ("FUNC_TABLE_GAP_PYTHON", "-DPython3_EXECUTABLE=")):
        if os.environ.get(env):
            args.append(flag + os.environ[env])
    print(f"  configuring {build_dir} from {source}")
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def build(build_dir: pathlib.Path) -> None:
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "kintera_release"],
                   check=True, stdout=subprocess.DEVNULL)


def installed_lib_dir() -> pathlib.Path | None:
    """Where the importable kintera keeps its shared libraries, asked in a THROWAWAY subprocess.

    Importing kintera here would mmap the very library this script swaps, and then the swap
    corrupts the loader's pages under a live mapping. That is what produced the exit-139
    segmentation fault at teardown: not a CUDA context outliving anything -- no CUDA call is
    needed to trigger it, and the probe already runs in its own process. A subprocess answers the
    question and takes its mapping with it when it exits.
    """
    r = subprocess.run([sys.executable, "-c",
                        "import kintera,pathlib;print(pathlib.Path(kintera.__file__).resolve().parent/'lib')"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    d = pathlib.Path(r.stdout.strip())
    return d if d.is_dir() else None


def swap_library(new_lib: pathlib.Path, dest: pathlib.Path) -> None:
    """Install `new_lib` at `dest` atomically.

    Never copy onto `dest` directly. A process that has it mapped keeps reading the same inode, so
    an in-place rewrite changes code under its feet; a rename gives it a new inode and leaves the
    old one intact until the last mapping goes away.
    """
    tmp = dest.with_name(dest.name + ".func_table_gap.swap")
    shutil.copy2(new_lib, tmp)
    os.replace(tmp, dest)


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


def cache_value(build_dir: pathlib.Path, key: str) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(key + ":"):
            return line.split("=", 1)[1].strip()
    return None


def fresh_configure(build_dir: pathlib.Path, source: pathlib.Path, reuse: bool) -> None:
    """Configure this build directory from scratch, for this source tree.

    A disposable checkout is normally made with `cp -a`, which brings the build directory along.
    Everything in it -- the cache, the generated makefiles, and a CMakeCache inside every
    FetchContent `_deps/*-subbuild` -- still names the ORIGINAL tree. `cmake --build` then prints
    "Build files have been written to: <original>/build-cuda", rebuilds nothing here, and exits 0,
    so the whole cycle reports success while the library is never rebuilt. Repairing that
    piecemeal is not worth it: the only state that is certainly ours is state we generated, so by
    default the directory is removed and configured again. `--reuse-build` skips this for a build
    directory you know was made here.
    """
    if reuse:
        print("  --reuse-build: configuring skipped; the build directory must already be this tree's")
        return
    if build_dir.exists():
        shutil.rmtree(build_dir)
    args = ["cmake", "-S", str(source.resolve()), "-B", str(build_dir.resolve()),
            "-DCMAKE_BUILD_TYPE=Release", "-DCUDA=ON", "-DNETCDF=ON"]
    for env, flag in (("CMAKE_CUDA_ARCHITECTURES", "-DCMAKE_CUDA_ARCHITECTURES="),
                      ("FUNC_TABLE_GAP_PYTHON", "-DPython3_EXECUTABLE=")):
        if os.environ.get(env):
            args.append(flag + os.environ[env])
    print(f"  configuring {build_dir} from {source}")
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def build(build_dir: pathlib.Path) -> None:
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "kintera_release"],
                   check=True, stdout=subprocess.DEVNULL)


def installed_lib_dir() -> pathlib.Path | None:
    """where the importable kintera keeps its shared libraries, if there is one"""
    try:
        import kintera
    except Exception:
        return None
    d = pathlib.Path(kintera.__file__).resolve().parent / "lib"
    return d if d.is_dir() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", required=True, help="a -DCUDA=ON Release build directory")
    ap.add_argument("--where", choices=["end", "front"], required=True)
    ap.add_argument("--names", default="h2o_ideal h2o_ideal_ddT nh3_ideal na_h2s_visscher_ddT")
    ap.add_argument("--dry-run", action="store_true", help="write the patched source and stop")
    ap.add_argument("--reuse-build", action="store_true",
                    help="do not wipe and reconfigure; only for a build directory made here")
    args = ap.parse_args()

    repo = pathlib.Path(os.environ.get("KINTERA_REPO", ""))
    if not repo.exists():
        sys.exit("func_table_gap.py: set KINTERA_REPO to a kintera checkout")
    src = repo / "src" / "func_table.cpp"
    build_dir = pathlib.Path(args.build)

    original = src.read_text()
    backup = src.with_suffix(".cpp.func_table_gap.bak")
    here = pathlib.Path(__file__).resolve().parent
    probe = here / "func_table_probe.py"
    site = installed_lib_dir()
    lib = build_dir / "lib" / "libkintera_release.so"
    saved_install = None
    baseline = None
    probe_failed = 0

    shutil.copy2(src, backup)
    try:
        if not args.dry_run:
            fresh_configure(build_dir, repo, args.reuse_build)
            build(build_dir)                      # baseline in THIS directory, for a fair digest
            baseline = sha(lib)
            print(f"baseline library in this build directory: {baseline}")
            if site is not None:
                saved_install = site / "libkintera_release.so.func_table_gap.bak"
                shutil.copy2(site / "libkintera_release.so", saved_install)

        src.write_text(patched_source(original, args.where))
        os.utime(src, None)
        print(f"patched {src} ({args.where}); CUDA table in src/func_table.cu untouched")
        if args.dry_run:
            return 0

        build(build_dir)
        mutated = sha(lib)
        print(f"rebuilt kintera_release (CPU library and its tables only): {mutated}")
        if mutated == baseline:
            # Rebuilding to the same bytes means the patch never reached the compiler. Reporting
            # "no difference" from that run would be a false negative, so stop instead.
            sys.exit("func_table_gap.py: the library is unchanged after patching, so nothing was "
                     "rebuilt. Refusing to report a result from a build that did not happen.")
        if site is not None:
            # Without this the probe imports the INSTALLED kintera and reads the old library, so the
            # mutation is invisible and the experiment silently reports agreement.
            swap_library(lib, site / "libkintera_release.so")
            print(f"  refreshed the installed binding at {site}")
        else:
            print("  WARNING: no importable kintera found; the probe will not see the mutation")
        rc = subprocess.run([sys.executable, "-P", str(probe), *args.names.split()]).returncode
        print(f"probe exit {rc}")
        if rc != 0:
            # The probe is the measurement. Printing its status and carrying on would turn a
            # failed experiment into a silent pass, which is the class of bug this script already
            # had once. Restoration still runs: it is in the finally below.
            probe_failed = rc
    finally:
        shutil.move(str(backup), str(src))
        # The backup keeps the original mtime, so the restored file looks OLDER than the object
        # built from the patched one and cmake skips it: the "restored" library is still the
        # mutated one, and the digest comparison below silently compares it with itself. Stamping
        # the file now is what makes the restore real.
        os.utime(src, None)
        if not args.dry_run and baseline is not None:
            build(build_dir)
            restored = sha(lib)
            same = restored == baseline
            print(f"restored {src} and rebuilt in the SAME build directory; "
                  f"{baseline} -> {restored} "
                  f"({'bit-identical' if same else 'NOT bit-identical -- investigate'})")
            if saved_install is not None:
                swap_library(saved_install, site / "libkintera_release.so")
                saved_install.unlink(missing_ok=True)
                print("  restored the installed binding")
    return probe_failed


if __name__ == "__main__":
    sys.exit(main())
