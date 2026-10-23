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
NEW = ["ei_case.py", "absent_gas_sweep.py", "func_table_gap.py", "func_table_probe.py"]
DRIVERS = RECOVERED + NEW


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


def test_ei_case_rejects_an_unknown_case():
    r = subprocess.run([sys.executable, "-P", str(RUNS / "ei_case.py"), "cuda", "nonsense"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "invalid choice" in r.stderr


def test_ei_case_offers_exactly_the_four_documented_cases():
    """`cuda ok`, `cuda cpu`, `cuda float32` and `cuda narrow` are what chapter 12 cites."""
    assert set(load("ei_case.py").CASES) == {"ok", "cpu", "float32", "narrow"}
def test_absent_gas_sweep_matches_the_documented_grid():
    m = load("absent_gas_sweep.py")
    assert m.MAX_ITER == 400 and m.FTOL == 1e-13
    assert min(m.TEMPS) == 110.0 and max(m.TEMPS) == 1600.0
    assert min(m.PRESSURES) == 1e2 and max(m.PRESSURES) == 1e7
    assert len(m.AMOUNTS) == 5


def test_absent_gas_variants_cover_none_one_and_two_zeroed():
    v = list(load("absent_gas_sweep.py").zeroed_variants(4))
    assert () in v
    assert sum(1 for x in v if len(x) == 1) == 4
    assert sum(1 for x in v if len(x) == 2) == 6
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
