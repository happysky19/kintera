"""The chapter-12 experiment drivers: argument handling, input construction, and refusals.

These import the drivers and exercise the parts that do not need kintera or a GPU, because that is
what can be checked anywhere. The numerical legs need a CUDA build at the chapter's pin and are run
by hand; what is tested here is that the drivers take the documented arguments, build the documented
inputs, and refuse rather than quietly succeed when a prerequisite is missing.
"""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

RUNS = pathlib.Path(__file__).resolve().parents[2] / "src" / "kintera_report" / "ch12" / "runs"
# Recovered from the report's own history at 516917515670447774cada7d52ba5cb705df4987.
RECOVERED = ["probe.py", "one_cell.py"]
# Written here; no original was found in the history or the searched workspace.
NEW = ["ei_case.py", "func_table_gap.py", "func_table_probe.py"]
# Authentic, recovered from a retained local bundle that is NOT a git checkout, so unlike
# RECOVERED above there is no producing commit to cite.
UNVERSIONED = ["absent_gas/run_sweep.py", "absent_gas/sweep.py", "absent_gas/k15_common.py"]
DRIVERS = RECOVERED + NEW + UNVERSIONED


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], RUNS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", DRIVERS)
def test_every_driver_is_present(name):
    assert (RUNS / name).exists(), f"{name} is cited by chapter 12 and must be committed"


@pytest.mark.parametrize("name", NEW)
def test_the_drivers_written_here_import_without_kintera(name):
    """The recovered two import kintera at module scope, as they always did, so they are not
    importable on a machine without it. That is kept rather than edited away."""
    load(name)


@pytest.mark.parametrize("name", NEW)
def test_every_driver_answers_help_without_kintera(name):
    """--help must work with no kintera and no GPU. Only for the drivers written here: the
    recovered ones are kept byte-for-byte as they were, so they are not given a CLI they never had."""
    r = subprocess.run([sys.executable, "-P", str(RUNS / name), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout.lower()
def test_func_table_gap_patches_the_cpu_table_and_leaves_the_cuda_table_alone():
    m = load("func_table_gap.py")
    source = (pathlib.Path(__file__).resolve().parents[3] / "src" / "func_table.cpp")
    text = source.read_text() if source.exists() else """
namespace kintera {
user_func1 func1_table_cpu[] = {
    h2o_ideal,       h2o_ideal_ddT,
    na_h2s_visscher, na_h2s_visscher_ddT};
std::vector<std::string> func1_names = {
    "h2o_ideal",          "h2o_ideal_ddT",
    "na_h2s_visscher",
    "na_h2s_visscher_ddT"};
}
"""
    for where in ("end", "front"):
        out = m.patched_source(text, where)
        assert m.TEST_FN in out
        assert "func1_table_cuda" not in out          # the device table is in func_table.cu
        assert out.count(m.TEST_FN) >= 4              # table entry, name, and the two definitions


def test_func_table_gap_refuses_to_patch_an_already_patched_source():
    m = load("func_table_gap.py")
    with pytest.raises(SystemExit):
        m.patched_source(f"namespace kintera {{ {m.TEST_FN} }}", "end")


def test_the_recovered_drivers_are_kept_verbatim():
    """Their value is that a rerun is comparable with the committed records, which only holds while
    the code is the code that produced them. Only the provenance header may differ."""
    for name in RECOVERED:
        text = (RUNS / name).read_text()
        assert "RECOVERED, not newly written" in text
        assert "516917515670447774cada7d52ba5cb705df4987" in text


def test_the_recovered_producer_keeps_its_campaign_constants():
    src = (RUNS / "probe.py").read_text()
    assert "SEED=20261020" in src.replace(" ", "")
    assert "'test_evaporation_kessler','test_coagulation_pressure_switch','jupiter'" in src.replace(" ", "")


def test_the_recovered_producer_measures_tolerance_against_the_cpu_side():
    """The saved records' outside_tolerance counts the CPU array as the reference. Measuring against
    the CUDA side instead changes the counts, which is what made an earlier reconstruction disagree
    with the committed float32 records."""
    src = (RUNS / "probe.py").read_text()
    assert "delta>atol+rtol*a.double().abs()" in src.replace(" ", "")


def test_one_cell_uses_the_recorded_single_cell_composition():
    src = (RUNS / "one_cell.py").read_text()
    assert "x[:,1:9]=.02" in src.replace(" ", "")
    assert "TEMP[1:2]" in src.replace(" ", "")


def test_the_absent_gas_grid_is_the_one_that_was_run():
    src = (RUNS / "absent_gas" / "run_sweep.py").read_text()
    assert "110,130,150,170,190,210,230,250,270,290,310,350,400,600,900,1200,1600" in src.replace(" ", "")
    assert "1e2,1e3,1e4,1e5,1e6,1e7" in src.replace(" ", "")
    assert src.count("1e-5") and "2.5e-3" in src


def test_the_absent_gas_bundle_admits_it_has_no_producing_commit():
    readme = (RUNS / "absent_gas" / "README.md").read_text()
    assert "no producing git SHA" in readme
    assert "102 instances did not converge" in readme or "102 did not" in readme


def test_func_table_gap_configures_fresh_by_default():
    """A copied build directory names the original tree everywhere, not just in CMakeCache.txt."""
    m = load("func_table_gap.py")
    src = (RUNS / "func_table_gap.py").read_text()
    assert "shutil.rmtree(build_dir)" in src
    assert "--reuse-build" in src


def test_func_table_gap_refreshes_the_installed_binding():
    """Without this the probe imports the installed kintera and never sees the mutation."""
    src = (RUNS / "func_table_gap.py").read_text()
    assert "installed_lib_dir" in src
    assert "libkintera_release.so" in src


def test_func_table_gap_refuses_a_build_that_did_not_change():
    src = (RUNS / "func_table_gap.py").read_text()
    assert "the library is unchanged after patching" in src


def test_func_table_gap_stamps_mtime_so_the_restore_really_rebuilds():
    """shutil.move keeps the old mtime, so without this the restored source looks older than the
    object built from the patched one and cmake skips it."""
    src = (RUNS / "func_table_gap.py").read_text()
    assert src.count("os.utime(src, None)") >= 2


def test_the_parent_never_imports_the_library_it_swaps():
    """Importing kintera in the parent mmaps the very file the cycle overwrites. Measured: an
    in-place overwrite after `import kintera` segfaults at teardown (exit 139); the same swap done
    atomically exits 0. The lookup therefore happens in a throwaway subprocess."""
    src = (RUNS / "func_table_gap.py").read_text()
    i = src.index("def installed_lib_dir")
    body = src[i:src.index("\ndef ", i + 1)]
    assert "subprocess.run" in body
    assert "import kintera" not in body.replace('"import kintera,pathlib;', "")


def test_library_swaps_are_atomic_not_in_place():
    """os.replace gives the running process a new inode; shutil.copy2 onto a mapped file does not."""
    src = (RUNS / "func_table_gap.py").read_text()
    i = src.index("def swap_library")
    body = src[i:src.index("\ndef ", i + 1)]
    assert "os.replace(tmp, dest)" in body
    # and the in-place overwrite is gone from the swap sites
    assert "shutil.copy2(lib, site" not in src


def test_a_failing_probe_is_propagated_not_printed_and_dropped():
    src = (RUNS / "func_table_gap.py").read_text()
    assert "probe_failed = rc" in src
    assert "return probe_failed" in src


def test_restoration_is_unconditional():
    """Restoration lives in finally, so a failing probe still puts the tree and binding back."""
    src = (RUNS / "func_table_gap.py").read_text()
    fin = src[src.index("    finally:"):]
    assert "shutil.move(str(backup), str(src))" in fin
    assert "os.utime(src, None)" in fin
    assert "swap_library(saved_install" in fin
