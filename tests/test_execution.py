from __future__ import annotations

import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest

from generate_mgxs import (
    Case,
    load_mgxs,
    load_opensn_result,
    prepare,
    run_openmc,
    run_opensn,
    solve_infinite_medium_eigenvalue,
)
from generate_mgxs.opensn import _convergence, _eigenvalue_convergence
from conftest import (
    OPENMC_DATA,
    OPENMC_PYTHON,
    OPENSN,
    OPENSN_FISSION_MGXS,
    OPENSN_MPI,
    _runtime_path,
    material,
    write_result,
    write_tiny_mgxs,
)


def eigenvalue_case(*, energy_groups=(1.0e-5, 1.0e6, 2.0e7)):
    """Return a fast homogeneous eigenvalue definition for runner tests."""
    return Case(
        name="tiny_eigenvalue",
        materials=(material(),),
        energy_groups=energy_groups,
        run_mode="eigenvalue",
        target_dimensions_cm=(2.0, 2.0, 2.0),
        batches=5,
        inactive_batches=1,
        particles_per_batch=10,
        scattering_order=0,
        keigen_tolerance=1.0e-8,
        keigen_max_iterations=50,
    )


def test_execution_runtime_paths_are_environment_configurable(monkeypatch, tmp_path):
    configured = tmp_path / "runtime"
    monkeypatch.setenv("GENERATE_MGXS_TEST_RUNTIME", str(configured))

    assert _runtime_path("GENERATE_MGXS_TEST_RUNTIME") == configured

    monkeypatch.delenv("GENERATE_MGXS_TEST_RUNTIME")
    assert _runtime_path(
        "GENERATE_MGXS_TEST_RUNTIME", default=configured
    ) == configured


def executable_script(path: Path, body: str):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path


def fake_openmc(path: Path, behavior="success"):
    if behavior == "failure":
        return executable_script(
            path,
            "import sys\n"
            "if sys.argv[1] == '-c': print('0.15.0')\n"
            "else:\n"
            "    assert sys.argv[1] == '-u'\n"
            "    print('failure stdout before exit', flush=True)\n"
            "    print('failure stderr before exit', file=sys.stderr, flush=True)\n"
            "    raise SystemExit(7)\n",
        )
    if behavior == "timeout":
        return executable_script(path, "import time; time.sleep(5)\n")
    process = "pass" if behavior == "missing" else '''
(cwd / "mgxs.h5").write_bytes(b"fixture")
(cwd / "openmc_result.json").write_text(json.dumps({"energy_bounds":[1,2],"flux":[1],"std_dev":[.1]}))
(cwd.parent / "diagnostics").mkdir(exist_ok=True)
(cwd.parent / "diagnostics/mgxs_uncertainty.json").write_text("{}")
'''
    return executable_script(path, f'''import json
import os
import sys
from pathlib import Path
if sys.argv[1] == "-c":
    print("0.15.0")
    raise SystemExit(0)
assert sys.argv[1] == "-u"
cwd = Path.cwd()
operation = sys.argv[3]
(cwd / f"{{operation}}_threads.txt").write_text(
    os.environ.get("MGXS_PROCESSES", "<missing>")
)
print(f"{{operation}} stdout", flush=True)
print(f"{{operation}} stderr", file=sys.stderr, flush=True)
if operation == "run":
    (cwd / "statepoint.2.h5").write_bytes(b"fixture")
elif operation == "process":
{''.join('    ' + line + chr(10) for line in process.splitlines())}
''')


def fake_opensn(
    path: Path,
    output="OpenSn version 1.0.1\nLinear Iteration 4 Residual 1.0e-9 status = converged\n",
    code=0,
):
    help_output = "OpenSn version 1.0.1" if "OpenSn version" in output else "unknown solver"
    return executable_script(path, f'''import sys
if "--help" in sys.argv:
    print({help_output!r})
    raise SystemExit(0)
print({output!r}, end="")
raise SystemExit({code})
''')


def successful_openmc_subprocess(calls):
    """Return a subprocess.run replacement that creates managed outputs."""
    def run(command, **kwargs):
        if command[1] == "-c":
            return subprocess.CompletedProcess(
                command, 0, stdout="0.15.0\n", stderr=""
            )

        phase = command[-1]
        calls.append(
            {
                "command": command,
                "environment": kwargs["env"],
                "stdout": Path(kwargs["stdout"].name).name,
                "stderr": Path(kwargs["stderr"].name).name,
            }
        )
        kwargs["stdout"].write(f"{phase} stdout\n")
        kwargs["stderr"].write(f"{phase} stderr\n")

        cwd = Path(kwargs["cwd"])
        if phase == "run":
            (cwd / "statepoint.2.h5").write_bytes(b"fixture")
        elif phase == "process":
            (cwd / "mgxs.h5").write_bytes(b"fixture")
            (cwd / "openmc_result.json").write_text(
                json.dumps(
                    {
                        "energy_bounds": [1, 2],
                        "flux": [1],
                        "std_dev": [0.1],
                    }
                )
            )
            (cwd.parent / "diagnostics").mkdir(exist_ok=True)
            (cwd.parent / "diagnostics/mgxs_uncertainty.json").write_text("{}")

        return subprocess.CompletedProcess(command, 0)

    return run


def test_openmc_success_and_required_outputs(one_case, tmp_path, monkeypatch):
    # OpenMC execution depends only on its generated model, so the helper must
    # work identically when OpenSn input was deliberately omitted.
    run = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>")

    calls = []
    monkeypatch.setattr(subprocess, "run", successful_openmc_subprocess(calls))
    monkeypatch.setenv("MGXS_PROCESSES", "99")

    result = run_openmc(
        run,
        cross_sections=cross_sections,
        python_executable=sys.executable,
    )

    assert result == run / "openmc/mgxs.h5"
    assert not (run / "opensn").exists()
    assert (run / "logs/openmc_run.stdout").read_text() == "run stdout\n"
    assert (run / "logs/openmc_run.stderr").read_text() == "run stderr\n"
    assert (run / "logs/openmc_process.stdout").read_text() == "process stdout\n"
    assert (run / "logs/openmc_process.stderr").read_text() == "process stderr\n"
    assert (run / "diagnostics/mgxs_uncertainty.json").is_file()
    assert [item["stdout"] for item in calls] == [
        "openmc_run.stdout",
        "openmc_process.stdout",
    ]
    assert [item["stderr"] for item in calls] == [
        "openmc_run.stderr",
        "openmc_process.stderr",
    ]
    assert all("MGXS_PROCESSES" not in item["environment"] for item in calls)

    metadata = json.loads((run / "_metadata/run.json").read_text())
    assert metadata["openmc"]["requested_threads"] is None
    assert metadata["openmc"]["commands"] == [
        [str(Path(sys.executable).resolve()), "-u", str(run / "openmc/model.py"), phase]
        for phase in ("run", "process")
    ]
    assert {item["path"] for item in metadata["artifacts"]} == {
        "diagnostics/mgxs_uncertainty.json",
        "openmc/mgxs.h5",
        "openmc/model.py",
        "openmc/openmc_result.json",
        "openmc/statepoint.2.h5",
    }


def test_openmc_explicit_threads_are_injected_exactly(
    one_case, tmp_path, monkeypatch
):
    run = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>")
    calls = []
    monkeypatch.setattr(subprocess, "run", successful_openmc_subprocess(calls))

    run_openmc(
        run,
        cross_sections=cross_sections,
        python_executable=sys.executable,
        operation="run",
        threads=6,
    )

    assert calls[0]["environment"]["MGXS_PROCESSES"] == "6"
    metadata = json.loads((run / "_metadata/run.json").read_text())["openmc"]
    assert metadata["requested_threads"] == 6


@pytest.mark.parametrize("threads", (0, -1, 1.5, "2", True, np.bool_(True)))
def test_openmc_rejects_invalid_explicit_threads(one_case, tmp_path, threads):
    run = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>")

    with pytest.raises(ValueError, match="threads must be an integer >= 1"):
        run_openmc(
            run,
            cross_sections=cross_sections,
            python_executable=fake_openmc(tmp_path / "openmc"),
            threads=threads,
        )


def test_generated_openmc_transport_uses_optional_thread_environment(
    one_case, tmp_path, monkeypatch
):
    run = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    generated = runpy.run_path(run / "openmc/model.py")
    calls = []

    class FakeModel:
        def run(self, *, threads):
            calls.append(threads)

    transport = generated["run_transport"]
    transport.__globals__["MODEL"] = FakeModel()

    monkeypatch.delenv("MGXS_PROCESSES", raising=False)
    transport()
    monkeypatch.setenv("MGXS_PROCESSES", "8")
    transport()

    assert calls == [None, 8]


def test_openmc_separate_phase_calls_preserve_each_others_logs(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>")
    executable = fake_openmc(tmp_path / "openmc")

    run_openmc(
        run,
        cross_sections=cross_sections,
        python_executable=executable,
        operation="run",
    )
    run_logs = {
        suffix: (run / f"logs/openmc_run.{suffix}").read_text()
        for suffix in ("stdout", "stderr")
    }

    run_openmc(
        run,
        cross_sections=cross_sections,
        python_executable=executable,
        operation="process",
    )
    assert {
        suffix: (run / f"logs/openmc_run.{suffix}").read_text()
        for suffix in ("stdout", "stderr")
    } == run_logs
    process_logs = {
        suffix: (run / f"logs/openmc_process.{suffix}").read_text()
        for suffix in ("stdout", "stderr")
    }

    run_openmc(
        run,
        cross_sections=cross_sections,
        python_executable=executable,
        operation="run",
    )
    assert {
        suffix: (run / f"logs/openmc_process.{suffix}").read_text()
        for suffix in ("stdout", "stderr")
    } == process_logs


def test_openmc_nonzero_exit(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    data = tmp_path / "cross_sections.xml"
    data.touch()

    with pytest.raises(subprocess.CalledProcessError):
        run_openmc(
            run,
            cross_sections=data,
            python_executable=fake_openmc(tmp_path / "bad", "failure"),
        )

    assert "failure stdout before exit" in (
        run / "logs/openmc_run.stdout"
    ).read_text()
    assert "failure stderr before exit" in (
        run / "logs/openmc_run.stderr"
    ).read_text()


def test_openmc_timeout(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    data = tmp_path / "cross_sections.xml"
    data.touch()

    with pytest.raises(subprocess.TimeoutExpired):
        run_openmc(
            run,
            cross_sections=data,
            python_executable=fake_openmc(tmp_path / "slow", "timeout"),
            timeout=0.01,
        )


def test_openmc_missing_executable_and_data(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="executable"):
        run_openmc(
            run,
            cross_sections=tmp_path / "data",
            python_executable=tmp_path / "missing",
        )

    executable = fake_openmc(tmp_path / "openmc")

    with pytest.raises(FileNotFoundError, match="cross_sections"):
        run_openmc(run, cross_sections=tmp_path / "data", python_executable=executable)


def test_openmc_missing_required_output(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "run")
    data = tmp_path / "cross_sections.xml"
    data.touch()

    with pytest.raises(FileNotFoundError, match="mgxs.h5"):
        run_openmc(
            run,
            cross_sections=data,
            python_executable=fake_openmc(tmp_path / "missing", "missing"),
            operation="process",
        )


@pytest.mark.skipif(
    not (OPENMC_PYTHON.is_file() and OPENMC_DATA.is_file()),
    reason="supplied OpenMC runtime is unavailable",
)
def test_generated_openmc_input_writes_native_model(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "real_openmc")
    run_openmc(
        run,
        cross_sections=OPENMC_DATA,
        python_executable=OPENMC_PYTHON,
        operation="write-input",
        timeout=30,
    )
    assert (run / "openmc/model.xml").is_file()


@pytest.mark.skipif(
    not (OPENMC_PYTHON.is_file() and OPENMC_DATA.is_file()),
    reason="supplied OpenMC runtime is unavailable",
)
def test_flattop_openmc_eigenvalue_input_writes_native_model(tmp_path):
    """The production FlatTop definition constructs native XML without transport."""
    from examples.flattop.case import CASE

    run = prepare(CASE, tmp_path / "flattop", solvers=("openmc",))
    run_openmc(
        run,
        cross_sections=OPENMC_DATA,
        python_executable=OPENMC_PYTHON,
        operation="write-input",
        timeout=30,
    )

    assert (run / "openmc/model.xml").is_file()


def prepared_fake_opensn(one_case, tmp_path, *, result=True):
    run = prepare(one_case, tmp_path / "run")
    write_tiny_mgxs(run / "openmc/mgxs.h5")
    if result:
        write_result(run / "opensn/opensn_result.json")
    return run


def test_opensn_rejects_openmc_only_preparation_before_execution(one_case, tmp_path):
    """Execution must not silently manufacture a solver input omitted by prepare()."""
    run_path = prepare(one_case, tmp_path / "run", solvers=("openmc",))

    with pytest.raises(FileNotFoundError, match="OpenSn input was not prepared"):
        run_opensn(run_path, executable=tmp_path / "never-used")


def test_opensn_success_requires_parsed_convergence(one_case, tmp_path):
    """A zero exit is accepted only with parseable explicit convergence."""
    run = prepared_fake_opensn(one_case, tmp_path)

    result = run_opensn(run, executable=fake_opensn(tmp_path / "opensn"))

    assert result.converged and result.iterations == 4
    assert result.residual == pytest.approx(1e-9)


def test_opensn_eigenvalue_convergence_requires_exact_final_summary():
    output = (
        "PI iteration = 8, k_eff = 1.0123457, k_eff_change = 8.00000e-09, "
        "status = converged\n"
        "PI final, status = converged, k_eff = 1.0123457, "
        "k_eff_change = 8.000000e-09, sweeps = 88\n"
    )

    parsed = _eigenvalue_convergence(
        output,
        1.0e-8,
        50,
        result_k_eff=1.01234569,
        result_iterations=8,
        result_sweeps=88,
    )

    assert parsed == pytest.approx((1.01234569, 8.0e-9, 8, 88))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            "PI iteration = 50, k_eff = 1.0, k_eff_change = 2e-7\n"
            "PI final, status = iteration_limit, k_eff = 1.0000000, "
            "k_eff_change = 2.000000e-07, sweeps = 500\n",
            "status=iteration_limit",
        ),
        (
            "PI iteration = 4, k_eff = 1.0, k_eff_change = 1e-9\n"
            "PI final, status = failed, k_eff = 1.0000000, "
            "k_eff_change = 1.000000e-09, sweeps = 40\n",
            "status=failed",
        ),
        (
            "PI iteration = 1, k_eff = 1.0, k_eff_change = 1e-9\n"
            "PI final, status = not_run, k_eff = 1.0000000, "
            "k_eff_change = 1.000000e-09, sweeps = 0\n",
            "status=not_run",
        ),
        ("PI iteration = 4, k_eff = 1.0, k_eff_change = 1e-9\n", "missing PI final"),
        (
            "PI iteration = 4, k_eff = 1.0, k_eff_change = 2e-7\n"
            "PI final, status = converged, k_eff = 1.0000000, "
            "k_eff_change = 2.000000e-07, sweeps = 40\n",
            "k_eff_change",
        ),
    ],
)
def test_opensn_eigenvalue_convergence_rejects_invalid_final_status(output, message):
    with pytest.raises(RuntimeError, match=message):
        _eigenvalue_convergence(
            output,
            1.0e-8,
            50,
            result_k_eff=1.0,
            result_iterations=4 if "iteration_limit" not in output else 50,
            result_sweeps=40 if "iteration_limit" not in output else 500,
        )


def test_opensn_eigenvalue_convergence_rejects_result_log_disagreement():
    output = (
        "PI iteration = 4, k_eff = 1.0000000, k_eff_change = 1e-9\n"
        "PI final, status = converged, k_eff = 1.0000000, "
        "k_eff_change = 1.000000e-09, sweeps = 40\n"
    )
    with pytest.raises(RuntimeError, match="k_eff disagrees"):
        _eigenvalue_convergence(
            output,
            1.0e-8,
            50,
            result_k_eff=1.001,
            result_iterations=4,
            result_sweeps=40,
        )


def test_opensn_eigenvalue_convergence_enforces_power_iteration_limit():
    output = (
        "PI iteration = 51, k_eff = 1.0000000, k_eff_change = 1e-9\n"
        "PI final, status = converged, k_eff = 1.0000000, "
        "k_eff_change = 1.000000e-09, sweeps = 510\n"
    )
    with pytest.raises(RuntimeError, match="configured limit"):
        _eigenvalue_convergence(
            output,
            1.0e-8,
            50,
            result_k_eff=1.0,
            result_iterations=51,
            result_sweeps=510,
        )


def test_run_opensn_loads_strict_eigenvalue_result(tmp_path):
    case = eigenvalue_case()
    run = prepare(case, tmp_path / "eigenvalue")
    write_tiny_mgxs(run / "openmc/mgxs.h5")
    result_path = run / "opensn/opensn_result.json"
    result_path.write_text(json.dumps({
        "run_mode": "eigenvalue",
        "energy_bounds": list(case.energy_bounds_ev),
        "flux": [0.25, 0.75],
        "logical_domain": "one",
        "solver": {
            "converged": None,
            "k_eff": 1.01234569,
            "k_eff_change": None,
            "power_iterations": 8,
            "sweeps": 88,
            "k_tolerance": 1.0e-8,
            "maximum_iterations": 50,
            "balance": 2.0e-12,
        },
    }))
    output = (
        "OpenSn version 1.0.1\n"
        "PI iteration = 8, k_eff = 1.0123457, k_eff_change = 8.00000e-09\n"
        "PI final, status = converged, k_eff = 1.0123457, "
        "k_eff_change = 8.000000e-09, sweeps = 88\n"
    )

    result = run_opensn(run, executable=fake_opensn(tmp_path / "opensn", output))

    assert result.run_mode == "eigenvalue"
    assert result.k_eff == pytest.approx(1.01234569)
    assert result.k_eff_change == pytest.approx(8.0e-9)
    assert result.power_iterations == 8
    assert result.sweeps == 88
    assert result.residual is None
    assert result.spectrum.values.sum() == pytest.approx(1.0)
    assert load_opensn_result(result_path).k_eff == pytest.approx(1.01234569)


def test_opensn_zero_return_nonconverged_is_rejected(one_case, tmp_path):
    """Meeting neither tolerance nor explicit status is a scientific failure."""
    run = prepared_fake_opensn(one_case, tmp_path)
    solver = fake_opensn(
        tmp_path / "opensn",
        "OpenSn version 1.0.1\nLinear Iteration 50 Residual 2.0e-8\n",
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        run_opensn(run, executable=solver)


def test_opensn_unknown_convergence_is_rejected(one_case, tmp_path):
    run = prepared_fake_opensn(one_case, tmp_path)
    with pytest.raises(RuntimeError, match="unknown"):
        run_opensn(
            run,
            executable=fake_opensn(tmp_path / "opensn", "OpenSn version 1.0.1\n"),
        )


def test_opensn_nonzero_and_timeout(one_case, tmp_path):
    run = prepared_fake_opensn(one_case, tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        run_opensn(run, executable=fake_opensn(tmp_path / "bad", code=3))

    assert "Linear Iteration" in (run / "logs/opensn.stdout").read_text()

    slow = executable_script(
        tmp_path / "slow",
        "import sys, time\n"
        "if '--help' in sys.argv: print('OpenSn version 1.0.1'); raise SystemExit(0)\n"
        "print('partial timeout log', flush=True)\n"
        "time.sleep(5)\n",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_opensn(run, executable=slow, timeout=0.5)

    # Direct streaming preserves partial diagnostics even when no CompletedProcess exists.
    assert "partial timeout log" in (run / "logs/opensn.stdout").read_text()


def test_opensn_missing_or_malformed_result(one_case, tmp_path):
    run = prepared_fake_opensn(one_case, tmp_path, result=False)
    executable = fake_opensn(tmp_path / "opensn")

    with pytest.raises(FileNotFoundError, match="result"):
        run_opensn(run, executable=executable)

    (run / "opensn/opensn_result.json").write_text("bad json")

    with pytest.raises(ValueError, match="malformed"):
        run_opensn(run, executable=executable)


def test_opensn_missing_identity_is_rejected(one_case, tmp_path):
    run = prepared_fake_opensn(one_case, tmp_path)
    executable = fake_opensn(tmp_path / "opensn", "Linear Iteration 4 Residual 1e-9\n")
    with pytest.raises(RuntimeError, match="identity"):
        run_opensn(run, executable=executable)


@pytest.mark.opensn
@pytest.mark.skipif(not OPENSN.is_file(), reason="supplied OpenSn console is unavailable")
def test_generated_one_material_opensn_serial(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "serial")
    write_tiny_mgxs(run / "openmc/mgxs.h5")

    result = run_opensn(run, executable=OPENSN, timeout=60)

    assert result.converged
    assert result.spectrum.values.shape == (2,)


@pytest.mark.opensn
@pytest.mark.skipif(
    not (OPENSN.is_file() and OPENSN_MPI.is_file()),
    reason="supplied OpenSn MPI runtime is unavailable",
)
def test_generated_one_material_opensn_two_rank(one_case, tmp_path):
    run = prepare(one_case, tmp_path / "mpi")
    write_tiny_mgxs(run / "openmc/mgxs.h5")
    result = run_opensn(
        run,
        executable=OPENSN,
        mpi_executable=OPENSN_MPI,
        ranks=2,
        timeout=60,
    )

    assert result.converged


@pytest.mark.opensn
@pytest.mark.skipif(
    not (OPENSN.is_file() and OPENSN_FISSION_MGXS.is_file()),
    reason="supplied OpenSn fissionable runtime fixture is unavailable",
)
def test_real_fissionable_direct_and_opensn_eigenvalues_agree(tmp_path):
    """The reflected P0 OpenSn verification solves the direct MGXS equation."""
    xs = load_mgxs(OPENSN_FISSION_MGXS, "set1", 294.0)
    direct = solve_infinite_medium_eigenvalue(xs)
    case = Case(
        name="u235_84g",
        materials=(material("set1"),),
        energy_groups=tuple(xs.energy_bounds_ev),
        run_mode="eigenvalue",
        target_dimensions_cm=(2.0, 2.0, 2.0),
        scattering_order=0,
        keigen_tolerance=1.0e-10,
        keigen_max_iterations=1000,
        batches=5,
        inactive_batches=1,
        particles_per_batch=10,
    )
    run = prepare(case, tmp_path / "real_eigenvalue")
    shutil.copyfile(OPENSN_FISSION_MGXS, run / "openmc/mgxs.h5")

    opensn = run_opensn(run, executable=OPENSN, timeout=60)
    difference = abs(opensn.k_eff - direct.k_eff)

    assert difference < 6.0e-8  # OpenSn's final log reports seven decimal places.
    assert opensn.spectrum.values.sum() == pytest.approx(1.0, abs=2.0e-14)
    np.testing.assert_array_equal(opensn.spectrum.energy_bounds_ev, xs.energy_bounds_ev)
    assert opensn.power_iterations <= case.keigen_max_iterations
