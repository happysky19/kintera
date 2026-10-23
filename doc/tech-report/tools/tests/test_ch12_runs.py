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
DRIVERS = ["probe.py", "ei_case.py", "ftol_sweep.py", "absent_gas_sweep.py",
           "kinetics_campaign.py", "func_table_gap.py"]


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], RUNS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", DRIVERS)
def test_every_driver_is_present_and_imports(name):
    assert (RUNS / name).exists(), f"{name} is cited by chapter 12 and must be committed"
    load(name)


@pytest.mark.parametrize("name", DRIVERS)
def test_every_driver_answers_help_without_kintera(name):
    """--help must work on a machine with no kintera and no GPU, or the driver cannot be inspected."""
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


def test_ei_case_uses_the_documented_cell_count_and_seed():
    m = load("ei_case.py")
    assert m.CELLS == 64
    assert m.SEED == 20261020


def test_kinetics_campaign_matches_the_documented_campaign_constants():
    m = load("kinetics_campaign.py")
    assert m.SEED == 20261020 and m.CELLS == 64 and m.DT == 10.0
    assert m.ATOL == 1e-12 and m.RTOL == 1e-10
    assert m.DECKS == ("test_evaporation_kessler", "test_coagulation_pressure_switch", "jupiter")


def test_ftol_sweep_matches_the_documented_state():
    m = load("ftol_sweep.py")
    assert m.CELLS == 200 and (m.T_LO, m.T_HI) == (250.0, 400.0)
    assert m.CELL == 1 and m.T_CELL == 250.75376892089844 and m.P_CELL == 1e5
    assert m.MAX_ITER == 50 and (m.ATOL, m.RTOL) == (1e-6, 1e-4)


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


def test_kinetics_campaign_refuses_to_write_without_a_destination():
    """It must never silently overwrite a saved record set."""
    r = subprocess.run([sys.executable, "-P", str(RUNS / "kinetics_campaign.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "--out" in (r.stdout + r.stderr)


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
