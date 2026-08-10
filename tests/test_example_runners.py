"""Fast control-flow tests for the best-effort example runners."""

from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).parents[1]


def _run_example(
    monkeypatch,
    tmp_path,
    example,
    *,
    failures=None,
    fail_first_mgxs_plot=False,
):
    failures = {} if failures is None else failures
    events = []
    captured_warnings = []
    openmc_spectrum = object()
    direct_spectrum = object()
    opensn_spectrum = object()
    direct_result = SimpleNamespace(k_eff=1.01, spectrum=direct_spectrum)
    opensn_result = SimpleNamespace(k_eff=1.02, spectrum=opensn_spectrum)
    if example in {"flattop", "pu9_hdpe"}:
        openmc_result = SimpleNamespace(
            k_eff=1.0,
            k_eff_std_dev=1.0e-4,
            spectrum=openmc_spectrum,
        )
    else:
        openmc_result = openmc_spectrum

    domains = tuple(
        SimpleNamespace(
            xsdata_name=f"domain_{index}",
            name=f"Domain {index}",
            material=SimpleNamespace(temperature_k=294.0),
        )
        for index in range(3)
    )
    case = SimpleNamespace(
        source_probabilities=(1.0,),
        source_volume_cm3=1.0,
        geometry=SimpleNamespace(domains=domains),
    )

    def record(name, result, *args, **kwargs):
        events.append((name, args, kwargs))
        if name in failures:
            raise failures[name]
        return result

    def prepare(*args, **kwargs):
        run = tmp_path / "run"
        (run / "openmc").mkdir(parents=True)
        return record("prepare", run, *args, **kwargs)

    def run_openmc(*args, **kwargs):
        operation = kwargs["operation"]
        events.append(("run_openmc", args, kwargs))
        failure = failures.get(f"run_openmc:{operation}")
        if failure is not None:
            raise failure
        if operation == "process":
            (tmp_path / "run/openmc/mgxs.h5").touch()
        return tmp_path / "run"

    mgxs_plot_calls = 0

    def plot_mgxs(*args, **kwargs):
        nonlocal mgxs_plot_calls
        mgxs_plot_calls += 1
        events.append(("plot_mgxs", args, kwargs))
        if fail_first_mgxs_plot and mgxs_plot_calls == 1:
            raise RuntimeError("first domain plot failed")
        if "plot_mgxs" in failures:
            raise failures["plot_mgxs"]
        return {"cross_sections": object()}

    api = ModuleType("generate_mgxs")
    api.prepare = prepare
    api.run_openmc = run_openmc
    api.load_openmc_result = lambda *args, **kwargs: record(
        "load_openmc_result", openmc_result, *args, **kwargs
    )
    api.load_mgxs = lambda *args, **kwargs: record(
        "load_mgxs", object(), *args, **kwargs
    )
    api.solve_infinite_medium = lambda *args, **kwargs: record(
        "solve_infinite_medium", direct_result, *args, **kwargs
    )
    api.solve_infinite_medium_eigenvalue = lambda *args, **kwargs: record(
        "solve_infinite_medium_eigenvalue", direct_result, *args, **kwargs
    )
    api.run_opensn = lambda *args, **kwargs: record(
        "run_opensn", opensn_result, *args, **kwargs
    )
    api.plot_mgxs = plot_mgxs
    api.plot_spectra = lambda *args, **kwargs: record(
        "plot_spectra", {"flux_spectrum": object()}, *args, **kwargs
    )
    api.load_openmc_domain_spectra = lambda *args, **kwargs: record(
        "load_openmc_domain_spectra",
        {domain.xsdata_name: object() for domain in domains},
        *args,
        **kwargs,
    )
    api.plot_openmc_domain_spectra = lambda *args, **kwargs: record(
        "plot_openmc_domain_spectra", {"domain_spectra": object()}, *args, **kwargs
    )

    case_module = ModuleType("case")
    case_module.CASE = case
    case_module.HDPE = SimpleNamespace(logical_name="hdpe", temperature_k=294.0)

    monkeypatch.setitem(sys.modules, "generate_mgxs", api)
    monkeypatch.setitem(sys.modules, "case", case_module)
    monkeypatch.setenv("OPENMC_CROSS_SECTIONS", "/fake/cross_sections.xml")
    monkeypatch.setenv("OPENSN_CONSOLE", "/fake/opensn-console")
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace = runpy.run_path(str(ROOT / "examples" / example / "run.py"))
        captured_warnings.extend(str(item.message) for item in caught)

    return SimpleNamespace(
        events=events,
        warnings=captured_warnings,
        namespace=namespace,
        openmc_spectrum=openmc_spectrum,
        direct_spectrum=direct_spectrum,
        opensn_spectrum=opensn_spectrum,
        domain_count=len(domains),
    )


def _event(result, name):
    return next(event for event in result.events if event[0] == name)


def test_flattop_direct_failure_continues_to_opensn_and_available_plots(
    monkeypatch, tmp_path, capsys
):
    result = _run_example(
        monkeypatch,
        tmp_path,
        "flattop",
        failures={
            "solve_infinite_medium_eigenvalue": ValueError(
                "eigenvalue loss operator is singular"
            )
        },
    )

    names = [event[0] for event in result.events]
    assert names.index("run_opensn") > names.index("solve_infinite_medium_eigenvalue")
    assert "plot_mgxs" in names
    spectrum_plot = _event(result, "plot_spectra")
    assert spectrum_plot[1][:3] == (
        result.openmc_spectrum,
        result.opensn_spectrum,
        None,
    )
    assert spectrum_plot[2]["include"] == ("openmc", "opensn")
    assert any(
        "Direct eigenvalue solve failed: eigenvalue loss operator is singular"
        in message
        for message in result.warnings
    )
    output = capsys.readouterr().out
    assert "Direct  k_eff = unavailable" in output
    assert "Direct eigenvalue solve     : FAILED" in output
    assert "OpenSn                      : PASSED" in output
    assert "Spectrum plots              : PASSED" in output


def test_opensn_failure_plots_only_openmc_and_direct(monkeypatch, tmp_path):
    result = _run_example(
        monkeypatch,
        tmp_path,
        "be9",
        failures={"run_opensn": RuntimeError("OpenSn did not converge")},
    )

    spectrum_plot = _event(result, "plot_spectra")
    assert spectrum_plot[1][:3] == (
        result.openmc_spectrum,
        None,
        result.direct_spectrum,
    )
    assert spectrum_plot[2]["include"] == ("openmc", "direct")
    assert any(
        "OpenSn failed: OpenSn did not converge" in message
        for message in result.warnings
    )


def test_failed_mgxs_plot_does_not_stop_spectrum_plotting(monkeypatch, tmp_path):
    result = _run_example(
        monkeypatch,
        tmp_path,
        "hdpe",
        failures={"plot_mgxs": RuntimeError("cannot render MGXS")},
    )

    assert _event(result, "plot_spectra")
    assert any(
        "MGXS plots failed: cannot render MGXS" in message
        for message in result.warnings
    )


def test_mgxs_load_failure_does_not_block_opensn_when_file_exists(
    monkeypatch, tmp_path
):
    result = _run_example(
        monkeypatch,
        tmp_path,
        "be9",
        failures={"load_mgxs": RuntimeError("Python MGXS load failed")},
    )

    names = [event[0] for event in result.events]
    assert "run_opensn" in names
    assert "solve_infinite_medium" not in names
    assert "plot_mgxs" not in names
    spectrum_plot = _event(result, "plot_spectra")
    assert spectrum_plot[1][:3] == (
        result.openmc_spectrum,
        result.opensn_spectrum,
        None,
    )
    assert spectrum_plot[2]["include"] == ("openmc", "opensn")


@pytest.mark.parametrize("example", ("detector", "pu9_hdpe"))
def test_multidomain_plot_failure_continues_remaining_domains(
    monkeypatch, tmp_path, example
):
    result = _run_example(
        monkeypatch,
        tmp_path,
        example,
        fail_first_mgxs_plot=True,
    )

    plot_calls = [event for event in result.events if event[0] == "plot_mgxs"]
    assert len(plot_calls) == result.domain_count
    assert _event(result, "plot_spectra")
    assert any(
        "MGXS plot (domain_0) failed: first domain plot failed" in message
        for message in result.warnings
    )


def test_processing_failure_skips_dependent_stages_without_dummy_results(
    monkeypatch, tmp_path
):
    result = _run_example(
        monkeypatch,
        tmp_path,
        "flattop",
        failures={"run_openmc:process": RuntimeError("statepoint is unreadable")},
    )

    names = [event[0] for event in result.events]
    assert "solve_infinite_medium_eigenvalue" not in names
    assert "run_opensn" not in names
    assert "plot_mgxs" not in names
    assert "plot_spectra" not in names
    assert any(
        "OpenMC processing failed: statepoint is unreadable" in message
        for message in result.warnings
    )
