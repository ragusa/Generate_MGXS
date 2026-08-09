from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from generate_mgxs import prepare, run_openmc, run_opensn
from conftest import (
    OPENMC_DATA,
    OPENMC_PYTHON,
    OPENSN,
    OPENSN_MPI,
    write_result,
    write_tiny_mgxs,
)


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
            "else: raise SystemExit(7)\n",
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
import sys
from pathlib import Path
if sys.argv[1] == "-c":
    print("0.15.0")
    raise SystemExit(0)
cwd = Path.cwd()
operation = sys.argv[2]
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


def test_openmc_success_and_required_outputs(one_case, tmp_path):
    # OpenMC execution depends only on its generated model, so the helper must
    # work identically when OpenSn input was deliberately omitted.
    run = prepare(one_case, tmp_path / "run", solvers=("openmc",))
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections/>")

    result = run_openmc(
        run,
        cross_sections=cross_sections,
        python_executable=fake_openmc(tmp_path / "openmc"),
    )

    assert result == run / "openmc/mgxs.h5"
    assert not (run / "opensn").exists()
    assert (run / "logs/openmc.stdout").is_file()
    assert (run / "diagnostics/mgxs_uncertainty.json").is_file()


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
@pytest.mark.skipif(not OPENSN.is_file(), reason="supplied OpenSn console is unavailable")
def test_generated_two_material_opensn_executes(two_case, tmp_path):
    run = prepare(two_case, tmp_path / "two")
    write_tiny_mgxs(run / "openmc/mgxs.h5", ("moderator", "target"))

    result = run_opensn(run, executable=OPENSN, timeout=60)

    assert set(result.domain_spectra) == {"target", "moderator"}

    document = json.loads((run / "opensn/opensn_result.json").read_text())
    assert document["domains"]["target"]["block"] == 0
    assert document["domains"]["target"]["volume_cm3"] == pytest.approx(0.064)
    assert document["domains"]["moderator"]["block"] == 1
