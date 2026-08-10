"""Generate, execute, and read explicit OpenSn calculations."""

from __future__ import annotations

from importlib import resources
import json
import os
from pathlib import Path
import pprint
import re
import subprocess

import numpy as np

from .case import Case, _artifact, _material_records, _update_run_metadata
from .openmc import _executable
from .results import OpenSnResult, Spectrum


def _write_opensn_input(case: Case, path: Path) -> None:
    """Render one deterministic, independently runnable OpenSn program."""
    materials = [
        {
            "logical_name": item["logical_name"],
            "temperature_k": item["temperature_k"],
            "role": item["role"],
        }
        for item in _material_records(case)
    ]

    numerical = {
        "run_mode": case.run_mode,
        "gmres_tolerance": case.gmres_tolerance,
        "gmres_max_iterations": case.gmres_max_iterations,
        "gmres_restart": case.gmres_restart,
        "keigen_tolerance": case.keigen_tolerance,
        "keigen_max_iterations": case.keigen_max_iterations,
    }

    # Keep the OpenSn API program in package data; this module only maps the
    # accepted Case into explicit, inspectable template values.
    text = (
        resources.files("generate_mgxs")
        .joinpath("templates/opensn_input.py.template")
        .read_text(encoding="utf-8")
    )

    replacements = {
        "__CASE_NAME__": repr(case.name),
        "__MATERIALS__": pprint.pformat(materials, width=100, sort_dicts=False),
        "__PHYSICAL_SOURCE__": pprint.pformat(
            case.source_definition, width=100, sort_dicts=False
        ),
        "__NUMERICAL__": pprint.pformat(numerical, width=100, sort_dicts=False),
        "__ENERGY_BOUNDS__": pprint.pformat(
            case.energy_bounds_ev, width=100, sort_dicts=False
        ),
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)

    path.write_text(text)


_ITERATION_RESIDUAL = re.compile(
    r"(?:linear\s+)?iteration\s*[:=]?\s*(\d+).*?residual(?:\s+norm)?\s*[:=]?\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_PETSC_RESIDUAL = re.compile(
    r"^\s*(\d+)\s+KSP\s+Residual\s+norm\s+"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_PI_ITERATION = re.compile(
    r"\bPI\s+iteration\s*=\s*(\d+)\s*,\s*k_eff\s*=\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*,\s*k_eff_change\s*=\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_PI_FINAL = re.compile(
    r"\bPI\s+final\s*,\s*status\s*=\s*([A-Za-z_]+)\s*,\s*"
    r"k_eff\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"k_eff_change\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"sweeps\s*=\s*(\d+)",
    re.IGNORECASE,
)


def _single_convergence(output: str, tolerance: float, maximum_iterations: int):
    """Validate one OpenSn solve from its persistent console output."""
    # OpenSn/PETSc versions emit two known residual formats.  A small residual
    # alone is insufficient: the log must also state explicit convergence.
    matches = _ITERATION_RESIDUAL.findall(output) or _PETSC_RESIDUAL.findall(output)
    if not matches:
        raise RuntimeError("OpenSn process succeeded but convergence is unknown")

    iterations, residual = int(matches[-1][0]), float(matches[-1][1])
    explicitly_failed = bool(
        re.search(
            r"\b(?:diverged|not\s+converged|converged\s*[:=]\s*false)\b",
            output,
            re.I,
        )
    )
    explicitly_converged = bool(
        re.search(r"status\s*[:=]\s*converged\b", output, re.I)
    )

    converged = (
        explicitly_converged
        and not explicitly_failed
        and residual <= tolerance
        and iterations <= maximum_iterations
    )

    if not converged:
        raise RuntimeError(
            f"OpenSn did not converge: iteration={iterations}, residual={residual:.8g}, "
            f"tolerance={tolerance:.8g}"
        )

    return iterations, residual


def _convergence(
    output: str,
    tolerance: float,
    maximum_iterations: int,
    domains=(),
):
    """Require explicit convergence for every independently solved domain."""
    if not domains:
        return _single_convergence(output, tolerance, maximum_iterations)

    metrics = []
    for domain in domains:
        begin = f"GENERATE_MGXS_DOMAIN_BEGIN {domain}"
        end = f"GENERATE_MGXS_DOMAIN_END {domain}"
        begin_index = output.find(begin)
        end_index = output.find(end, begin_index + len(begin))
        if (
            begin_index < 0
            or end_index < 0
            or output.count(begin) != 1
            or output.count(end) != 1
        ):
            raise RuntimeError(
                f"OpenSn convergence log is incomplete for domain {domain!r}"
            )
        segment = output[begin_index:end_index]
        metrics.append(_single_convergence(segment, tolerance, maximum_iterations))

    # The result-level summary is deliberately conservative; each domain has
    # already passed the same explicit status, residual, and iteration checks.
    return max(item[0] for item in metrics), max(item[1] for item in metrics)


def _eigenvalue_convergence(
    output: str,
    tolerance: float,
    maximum_iterations: int,
    *,
    result_k_eff: float,
    result_iterations: int,
    result_sweeps: int,
):
    """Require OpenSn's final power-iteration convergence record."""
    finals = _PI_FINAL.findall(output)
    iterations = _PI_ITERATION.findall(output)
    if len(finals) != 1 or not iterations:
        raise RuntimeError("OpenSn eigenvalue convergence is unknown: missing PI final")

    status, parsed_k, parsed_change, parsed_sweeps = finals[0]
    parsed_k = float(parsed_k)
    parsed_change = float(parsed_change)
    parsed_sweeps = int(parsed_sweeps)
    logged_iterations = int(iterations[-1][0])

    if status.lower() != "converged":
        raise RuntimeError(f"OpenSn eigenvalue did not converge: status={status}")
    if parsed_change > tolerance:
        raise RuntimeError(
            "OpenSn eigenvalue did not converge: "
            f"k_eff_change={parsed_change:.8g}, tolerance={tolerance:.8g}"
        )
    if result_iterations < 1 or result_iterations > maximum_iterations:
        raise RuntimeError(
            "OpenSn eigenvalue power-iteration count exceeds its configured limit"
        )
    if logged_iterations != result_iterations:
        raise RuntimeError("OpenSn eigenvalue iteration count disagrees with its log")
    if parsed_sweeps != result_sweeps:
        raise RuntimeError("OpenSn eigenvalue sweep count disagrees with its log")
    if not np.isclose(parsed_k, result_k_eff, rtol=0.0, atol=5.1e-8):
        # OpenSn formats the final value with seven digits after the decimal.
        raise RuntimeError("OpenSn eigenvalue result k_eff disagrees with its log")

    # Preserve the full-precision value written by GetEigenvalue(); the parsed
    # value is only independent convergence evidence at log precision.
    return result_k_eff, parsed_change, result_iterations, parsed_sweeps


def run_opensn(
    run_directory,
    *,
    executable,
    mpi_executable=None,
    ranks=1,
    timeout=None,
) -> OpenSnResult:
    """Run generated OpenSn input, preserving logs and requiring convergence."""
    # Validate prepared inputs before constructing a serial or MPI command.
    run = Path(run_directory).resolve()
    input_path = run / "opensn" / "input.py"
    mgxs = run / "openmc" / "mgxs.h5"
    if not input_path.is_file():
        raise FileNotFoundError("OpenSn input was not prepared for this run")
    if not mgxs.is_file():
        raise FileNotFoundError(f"OpenMC MGXS HDF5 is missing: {mgxs}")

    console = _executable(executable)
    if ranks < 1:
        raise ValueError("ranks must be positive")

    if ranks == 1:
        command = [console, "-i", str(input_path)]
    else:
        if mpi_executable is None:
            raise ValueError("mpi_executable is required when ranks > 1")
        command = [
            _executable(mpi_executable),
            "-n",
            str(ranks),
            console,
            "-i",
            str(input_path),
        ]

    # Query the console actually used for this run. OpenSn 1.0.1 prints valid
    # identity/help text even though its --help command returns nonzero.
    identity = subprocess.run(
        [console, "--help"], capture_output=True, text=True, timeout=timeout
    )
    version = re.search(r"OpenSn version\s+([^\s]+)", identity.stdout + identity.stderr)
    if not version:
        raise RuntimeError("OpenSn software identity could not be established")

    commit_match = re.search(r"commit-([0-9a-f]+)", console)
    opensn_metadata = {
        "version": version.group(1),
        "commit": commit_match.group(1) if commit_match else None,
        "executable": console,
        "command": command,
        "mpi_ranks": ranks,
    }
    _update_run_metadata(run, opensn=opensn_metadata)

    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "opensn.stdout"
    stderr_path = logs / "opensn.stderr"

    # Stream directly to disk so timeout, crash, and nonconvergence diagnostics
    # survive without buffering an arbitrarily large transport log in memory.
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            command,
            cwd=input_path.parent,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
        )

    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)

    # Parse the persistent files after the handles close so all solver output,
    # including final PETSc convergence lines, is visible.
    output = stdout_path.read_text() + stderr_path.read_text()
    result_path = run / "opensn" / "opensn_result.json"
    if not result_path.is_file():
        raise FileNotFoundError("OpenSn completed without writing opensn_result.json")

    try:
        document = json.loads(result_path.read_text())
        solver = document["solver"]
        run_mode = document.get("run_mode", "fixed_source")
        domains = document.get("domains", {})
        if not isinstance(domains, dict):
            raise TypeError("domains must be an object")
        if run_mode == "eigenvalue":
            tolerance = float(solver["k_tolerance"])
            maximum = int(solver["maximum_iterations"])
            result_k_eff = float(solver["k_eff"])
            result_iterations = int(solver["power_iterations"])
            result_sweeps = int(solver["sweeps"])
        elif run_mode == "fixed_source":
            tolerance = float(solver["tolerance"])
            maximum = int(solver["maximum_iterations"])
        else:
            raise ValueError("unknown run_mode")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("OpenSn result is malformed") from error

    if run_mode == "eigenvalue":
        parsed_k, change, iterations, sweeps = _eigenvalue_convergence(
            output,
            tolerance,
            maximum,
            result_k_eff=result_k_eff,
            result_iterations=result_iterations,
            result_sweeps=result_sweeps,
        )
        solver.update(
            converged=True,
            k_eff=parsed_k,
            k_eff_change=change,
            power_iterations=iterations,
            sweeps=sweeps,
        )
    else:
        # A low residual is accepted only when the log also reports explicit
        # convergence within the configured iteration limit.
        iterations, residual = _convergence(
            output, tolerance, maximum, tuple(domains)
        )
        solver.update(converged=True, iterations=iterations, residual=residual)

    result_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    artifacts = [_artifact(input_path, run), _artifact(mgxs, run), _artifact(result_path, run)]
    _update_run_metadata(run, artifacts=artifacts)

    return load_opensn_result(result_path)


def load_opensn_result(path) -> OpenSnResult:
    """Load an ascending-energy OpenSn result and reject nonconvergence."""
    path = Path(path)
    if path.is_dir():
        path = path / "opensn" / "opensn_result.json"

    document = json.loads(path.read_text())
    solver = document.get("solver", {})
    converged = solver.get("converged")
    run_mode = document.get("run_mode", "fixed_source")
    if converged is not True:
        raise RuntimeError("OpenSn result does not explicitly report convergence")

    # Generated OpenSn files reverse solver-native high-to-low fields before
    # serialization, so result loading retains the package's ascending order.
    bounds = np.asarray(document["energy_bounds"], dtype=float)
    primary = Spectrum(
        bounds,
        np.asarray(document["flux"], dtype=float),
        logical_domain=document.get("logical_domain"),
    )
    domains = None
    if "domains" in document:
        domains = {
            name: Spectrum(
                bounds,
                np.asarray(record["flux"], dtype=float),
                logical_domain=name,
            )
            for name, record in document["domains"].items()
        }

    balance = float(solver["balance"]) if solver.get("balance") is not None else None
    if run_mode == "eigenvalue":
        required = ("k_eff", "k_eff_change", "power_iterations", "sweeps")
        if any(solver.get(name) is None for name in required):
            raise RuntimeError(
                "OpenSn eigenvalue result does not explicitly report convergence"
            )
        if np.any(primary.values < 0.0) or not np.isclose(
            primary.values.sum(), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("OpenSn eigenvalue spectrum must be nonnegative and normalized")
        return OpenSnResult(
            primary,
            True,
            balance=balance,
            domain_spectra=domains,
            run_mode="eigenvalue",
            k_eff=solver["k_eff"],
            k_eff_change=solver["k_eff_change"],
            power_iterations=solver["power_iterations"],
            sweeps=solver["sweeps"],
        )

    iterations = solver.get("iterations", solver.get("iteration_count"))
    residual = solver.get("residual", solver.get("final_residual"))
    if iterations is None or residual is None:
        raise RuntimeError("OpenSn result does not explicitly report convergence")
    return OpenSnResult(
        primary,
        True,
        int(iterations),
        float(residual),
        balance,
        domains,
    )
