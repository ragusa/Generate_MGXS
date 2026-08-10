"""Generate, execute, and read explicit OpenMC calculations."""

from __future__ import annotations

from hashlib import sha256
from importlib import resources
import json
import os
from pathlib import Path
import pprint
import shutil
import subprocess
import sys

import numpy as np

from .case import Case, _artifact, _material_records, _update_run_metadata
from .results import OpenMCEigenvalueResult, Spectrum


def _write_openmc_input(case: Case, path: Path) -> None:
    """Render one deterministic, independently runnable OpenMC program."""
    materials = _material_records(case)

    geometry = {
        "type": case.geometry_type,
        "target_dimensions_cm": case.target_dimensions_cm,
        "outer_dimensions_cm": case.outer_dimensions_cm or case.target_dimensions_cm,
        "boundaries": dict(
            zip(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), case.boundaries)
        ),
    }

    history = {
        "run_mode": case.run_mode,
        "particles_per_batch": case.particles_per_batch,
        "batches": case.batches,
        "inactive_batches": case.inactive_batches,
        "total_histories": case.total_histories,
    }

    if case.energy_group_structure is None:
        explicit_bounds = pprint.pformat(
            case.energy_bounds_ev, width=100, sort_dicts=False
        )
        energy_group_definition = (
            "ENERGY_GROUP_STRUCTURE = None\n"
            f"ENERGY_BOUNDS_EV = np.asarray({explicit_bounds}, dtype=float)\n"
            "groups = openmc.mgxs.EnergyGroups(group_edges=ENERGY_BOUNDS_EV)"
        )
    else:
        # Do not serialize OpenMC's boundary table. The generated model asks
        # OpenMC to resolve the same canonical name used during preparation.
        structure = json.dumps(case.energy_group_structure)
        energy_group_definition = (
            f"ENERGY_GROUP_STRUCTURE = {structure}\n"
            f"groups = openmc.mgxs.EnergyGroups(group_edges={structure})\n"
            "ENERGY_BOUNDS_EV = groups.group_edges"
        )

    # The template is package data so this generator stays focused on mapping
    # Case values into a readable standalone solver program.
    text = (
        resources.files("generate_mgxs")
        .joinpath("templates/openmc_model.py.template")
        .read_text(encoding="utf-8")
    )

    replacements = {
        "__CASE_NAME__": repr(case.name),
        "__MATERIALS__": pprint.pformat(materials, width=100, sort_dicts=False),
        "__GEOMETRY__": pprint.pformat(geometry, width=100, sort_dicts=False),
        "__PHYSICAL_SOURCE__": pprint.pformat(
            case.source_definition, width=100, sort_dicts=False
        ),
        "__HISTORY__": pprint.pformat(history, width=100, sort_dicts=False),
        "__MGXS_SETTINGS__": pprint.pformat(
            {"scattering_order": case.scattering_order}, width=100, sort_dicts=False
        ),
        "__ENERGY_GROUP_DEFINITION__": energy_group_definition,
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)

    path.write_text(text)


def _executable(value) -> str:
    value = os.fspath(value)
    resolved = shutil.which(value) if not os.path.isabs(value) else value

    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"executable not found: {value}")

    return str(Path(resolved).resolve())


def run_openmc(
    run_directory,
    *,
    cross_sections,
    python_executable=sys.executable,
    operation="all",
    threads=1,
    timeout=None,
) -> Path:
    """Run the generated OpenMC input, capture logs, and verify its outputs."""
    # Validate every run-relative input before querying or starting OpenMC.
    run = Path(run_directory).resolve()
    model = run / "openmc" / "model.py"
    if not model.is_file():
        raise FileNotFoundError(f"generated OpenMC input is missing: {model}")

    python = _executable(python_executable)
    cross_sections = Path(cross_sections).resolve()
    if not cross_sections.is_file():
        raise FileNotFoundError(f"OpenMC cross_sections.xml is missing: {cross_sections}")

    if operation not in {"write-input", "run", "process", "all"}:
        raise ValueError("operation must be write-input, run, process, or all")

    # Record the actual OpenMC runtime rather than inferring it from the Python
    # executable path or the environment used during preparation.
    identity = subprocess.run(
        [python, "-c", "import openmc; print(openmc.__version__)"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if identity.returncode or not identity.stdout.strip():
        raise RuntimeError("OpenMC software identity could not be established")

    operations = ("run", "process") if operation == "all" else (operation,)
    commands = [[python, str(model), item] for item in operations]

    openmc_metadata = {
        "version": identity.stdout.strip().splitlines()[-1],
        "python": python,
        "nuclear_data": {
            "cross_sections": str(cross_sections),
            "bytes": cross_sections.stat().st_size,
            "sha256": sha256(cross_sections.read_bytes()).hexdigest(),
        },
        "commands": commands,
    }
    _update_run_metadata(run, openmc=openmc_metadata)

    # Give each phase persistent logs of its own. A later processing call must
    # not erase transport diagnostics, and either phase may retain partial
    # output when its subprocess fails.
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(python).parent) + os.pathsep + environment.get("PATH", "")
    )
    environment["OPENMC_CROSS_SECTIONS"] = str(cross_sections)
    environment["MGXS_PROCESSES"] = str(threads)

    for phase, command in zip(operations, commands):
        log_phase = phase.replace("-", "_")
        with (
            (logs / f"openmc_{log_phase}.stdout").open("w") as stdout,
            (logs / f"openmc_{log_phase}.stderr").open("w") as stderr,
        ):
            subprocess.run(
                command,
                cwd=model.parent,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout,
                check=True,
            )

    # A successful process exit is not enough: require the scientific products
    # promised by the selected operation.
    if operation in {"run", "all"}:
        statepoints = list((run / "openmc").glob("statepoint.*.h5"))
        if not statepoints:
            raise FileNotFoundError("OpenMC completed without producing a statepoint")
    if operation in {"process", "all"}:
        for required in ("mgxs.h5", "openmc_result.json"):
            if not (run / "openmc" / required).is_file():
                raise FileNotFoundError(f"OpenMC processing did not produce {required}")
        if not (run / "diagnostics" / "mgxs_uncertainty.json").is_file():
            raise FileNotFoundError("OpenMC processing did not preserve MGXS uncertainty")

    artifacts = []
    artifact_paths = [
        run / "openmc" / "model.py",
        run / "openmc" / "mgxs.h5",
        run / "openmc" / "openmc_result.json",
        run / "diagnostics" / "mgxs_uncertainty.json",
    ] + sorted((run / "openmc").glob("statepoint.*.h5"))
    for path in artifact_paths:
        if path.is_file():
            artifacts.append(_artifact(path, run))

    _update_run_metadata(run, artifacts=artifacts)

    return run / "openmc" / "mgxs.h5"


def load_openmc_result(path) -> Spectrum | OpenMCEigenvalueResult:
    """Load a compact fixed-source spectrum or OpenMC eigenvalue result."""
    path = Path(path)
    if path.is_dir():
        path = path / "openmc" / "openmc_result.json"

    document = json.loads(path.read_text())
    if "std_dev" not in document:
        raise ValueError("OpenMC result does not contain statistical uncertainty")

    spectrum = Spectrum(
        np.asarray(document["energy_bounds"], dtype=float),
        np.asarray(document["flux"], dtype=float),
        np.asarray(document["std_dev"], dtype=float),
        document.get("logical_domain"),
    )
    if document.get("run_mode") == "eigenvalue":
        try:
            return OpenMCEigenvalueResult(
                spectrum,
                document["k_eff"],
                document["k_eff_std_dev"],
            )
        except KeyError as error:
            raise ValueError("OpenMC eigenvalue result is missing k_eff data") from error

    return spectrum
