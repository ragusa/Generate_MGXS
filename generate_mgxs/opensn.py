"""Generate, execute, and read explicit OpenSn calculations."""

from __future__ import annotations

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
    materials = [
        {
            "logical_name": item["logical_name"],
            "temperature_k": item["temperature_k"],
            "role": item["role"],
            "opensn_block": item["opensn_block"],
            "openmc_id": item["openmc_id"],
        }
        for item in _material_records(case)
    ]
    geometry = {
        "type": case.geometry_type,
        "target_dimensions_cm": case.target_dimensions_cm,
        "outer_dimensions_cm": case.outer_dimensions_cm or case.target_dimensions_cm,
        "boundaries": dict(zip(
            ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
            ("reflecting" if item == "reflective" else item for item in case.boundaries),
        )),
        "source_volume_cm3": case.source_volume_cm3,
    }
    numerical = {
        "mesh_max_width_cm": case.mesh_max_width_cm,
        "num_polar": case.num_polar,
        "num_azimuthal": case.num_azimuthal,
        "scattering_order": case.scattering_order,
        "gmres_tolerance": case.gmres_tolerance,
        "gmres_max_iterations": case.gmres_max_iterations,
        "gmres_restart": case.gmres_restart,
    }
    text = '''\
"""Generated OpenSn input using the run-relative OpenMC MGXS HDF5 handoff.

Normally invoke as ``opensn-console -i input.py``.  This file reads
``../openmc/mgxs.h5`` and writes ``opensn_result.json`` beside itself.
"""

import json
import math
from pathlib import Path

import numpy as np


# --- Human-readable case definition ---------------------------------------
CASE_NAME = __CASE_NAME__
MATERIALS = __MATERIALS__

# Material roles fix block 0 as homogeneous/target and block 1 as moderator.
GEOMETRY = __GEOMETRY__

# This is the same physical authority used to build the continuous OpenMC source.
PHYSICAL_SOURCE = __PHYSICAL_SOURCE__

# Mesh, quadrature, scattering, and strict GMRES controls.
OPENSN_NUMERICAL_SETTINGS = __NUMERICAL__


# --- Energy-group data and run-relative paths -----------------------------
# Canonical Python data are ascending; OpenSn group zero is highest energy.
ENERGY_BOUNDS_EV_ASCENDING = np.asarray(__ENERGY_BOUNDS__, dtype=float)
RUN_DIRECTORY = Path(__file__).resolve().parents[1]
MGXS_HDF5 = RUN_DIRECTORY / "openmc" / "mgxs.h5"
RESULT_PATH = RUN_DIRECTORY / "opensn" / "opensn_result.json"
NUM_GROUPS = ENERGY_BOUNDS_EV_ASCENDING.size - 1

CONFIG = {
    "case_name": CASE_NAME,
    "materials": MATERIALS,
    "geometry_type": GEOMETRY["type"],
    "target_dimensions_cm": GEOMETRY["target_dimensions_cm"],
    "outer_dimensions_cm": GEOMETRY["outer_dimensions_cm"],
    "boundaries": GEOMETRY["boundaries"],
    "source_volume_cm3": GEOMETRY["source_volume_cm3"],
    **OPENSN_NUMERICAL_SETTINGS,
}


def source_probabilities():
    """Derive integrated group masses in ascending physical-energy order."""
    kind = PHYSICAL_SOURCE["kind"]
    if kind == "grouped":
        masses = np.asarray(PHYSICAL_SOURCE["probabilities"], dtype=float)
        if masses.shape != (NUM_GROUPS,) or np.any(masses < 0.0):
            raise ValueError("source must provide one nonnegative mass per group")
        if not np.isclose(masses.sum(), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("explicit group probabilities must sum to one")
        return masses
    elif kind == "uniform_energy":
        masses = np.diff(ENERGY_BOUNDS_EV_ASCENDING)
    elif kind == "watt":
        bounds_mev = ENERGY_BOUNDS_EV_ASCENDING / 1.0e6
        nodes, weights = np.polynomial.legendre.leggauss(64)
        masses = []
        for low, high in zip(bounds_mev[:-1], bounds_mev[1:]):
            energies = 0.5 * (high - low) * nodes + 0.5 * (high + low)
            density = np.exp(-energies / PHYSICAL_SOURCE["a_mev"]) * np.sinh(
                np.sqrt(PHYSICAL_SOURCE["b_per_mev"] * energies)
            )
            masses.append(0.5 * (high - low) * np.dot(weights, density))
        masses = np.asarray(masses)
    else:
        raise ValueError(f"unsupported physical source kind: {kind}")
    if masses.shape != (NUM_GROUPS,) or np.any(masses < 0.0):
        raise ValueError("source must provide one nonnegative mass per group")
    masses = masses / masses.sum()
    masses[-1] += 1.0 - masses.sum()
    return masses


SOURCE_PROBABILITIES_ASCENDING = source_probabilities()


def subdivide(low, high, max_width):
    """Subdivide one mesh interval without exceeding max_width."""
    count = max(1, math.ceil((high - low) / max_width))
    return np.linspace(low, high, count + 1).tolist()


def axis_nodes(outer_length, target_length, max_width):
    """Create nodes that exactly retain target interfaces."""
    outside = (-outer_length / 2, outer_length / 2)
    target = (-target_length / 2, target_length / 2)
    if math.isclose(outer_length, target_length, abs_tol=1e-12):
        return subdivide(*outside, max_width)
    nodes = []
    for low, high in ((outside[0], target[0]), target, (target[1], outside[1])):
        segment = subdivide(low, high, max_width)
        nodes.extend(segment if not nodes else segment[1:])
    return nodes


# --- Mesh and material/block mapping --------------------------------------
target_dimensions = tuple(CONFIG["target_dimensions_cm"])
outer_dimensions = tuple(CONFIG["outer_dimensions_cm"])
nodes = [
    axis_nodes(outer, target, width)
    for outer, target, width in zip(
        outer_dimensions, target_dimensions, CONFIG["mesh_max_width_cm"]
    )
]
grid = OrthogonalMeshGenerator(node_sets=nodes).Execute()
whole_volume = RPPLogicalVolume(
    xmin=-outer_dimensions[0] / 2, xmax=outer_dimensions[0] / 2,
    ymin=-outer_dimensions[1] / 2, ymax=outer_dimensions[1] / 2,
    zmin=-outer_dimensions[2] / 2, zmax=outer_dimensions[2] / 2,
)
target_volume = RPPLogicalVolume(
    xmin=-target_dimensions[0] / 2, xmax=target_dimensions[0] / 2,
    ymin=-target_dimensions[1] / 2, ymax=target_dimensions[1] / 2,
    zmin=-target_dimensions[2] / 2, zmax=target_dimensions[2] / 2,
)
if CONFIG["geometry_type"] == "homogeneous":
    grid.SetUniformBlockID(0)
else:
    grid.SetUniformBlockID(1)
    grid.SetBlockIDFromLogicalVolume(target_volume, 0, True)

measured_volumes = {
    int(block): float(volume) for block, volume in grid.ComputeVolumePerBlockID().items()
}
expected_volumes = {0: CONFIG["source_volume_cm3"]}
if CONFIG["geometry_type"] == "moderated_target":
    expected_volumes[1] = math.prod(outer_dimensions) - CONFIG["source_volume_cm3"]
for block, expected in expected_volumes.items():
    if block not in measured_volumes or not math.isclose(
        measured_volumes[block], expected, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError(f"OpenSn block {block} has the wrong physical volume")

# Every material explicitly names its HDF5 dataset and derived OpenSn block.
# Neither object ordering nor the diagnostic OpenMC ID controls this mapping.
if not MGXS_HDF5.is_file():
    raise FileNotFoundError(f"OpenMC MGXS HDF5 is missing: {MGXS_HDF5}")
cross_sections = []
xs_by_domain = {}
for material in CONFIG["materials"]:
    xs = MultiGroupXS()
    xs.LoadFromOpenMC(
        str(MGXS_HDF5), material["logical_name"], material["temperature_k"]
    )
    cross_sections.append({"block_ids": [material["opensn_block"]], "xs": xs})
    xs_by_domain[material["logical_name"]] = xs

# --- Angular discretization and transport solver --------------------------
quadrature = GLCProductQuadrature3DXYZ(
    n_polar=CONFIG["num_polar"],
    n_azimuthal=CONFIG["num_azimuthal"],
    scattering_order=CONFIG["scattering_order"],
)
groupsets = [{
    "groups_from_to": (0, NUM_GROUPS - 1),
    "angular_quadrature": quadrature,
    "inner_linear_method": "petsc_gmres",
    "l_abs_tol": CONFIG["gmres_tolerance"],
    "l_max_its": CONFIG["gmres_max_iterations"],
    "gmres_restart_interval": CONFIG["gmres_restart"],
}]
problem = DiscreteOrdinatesProblem(
    mesh=grid,
    num_groups=NUM_GROUPS,
    groupsets=groupsets,
    xs_map=cross_sections,
    boundary_conditions=[
        {"name": name, "type": CONFIG["boundaries"][name]}
        for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    ],
    options={"save_angular_flux": False},
)

if SOURCE_PROBABILITIES_ASCENDING.shape != (NUM_GROUPS,):
    raise ValueError("source must contain one integrated probability per group")
if np.any(SOURCE_PROBABILITIES_ASCENDING < 0.0) or not math.isclose(
    float(SOURCE_PROBABILITIES_ASCENDING.sum()), 1.0, abs_tol=1e-12
):
    raise ValueError("source group probabilities must be nonnegative and sum to one")
# OpenSn numbers group 0 at highest energy, so reverse the ascending physical
# source vector at this boundary only. Division by target volume converts one
# source particle to the spatial source density required by the equation.
source_strength_high_to_low = (
    SOURCE_PROBABILITIES_ASCENDING[::-1] / CONFIG["source_volume_cm3"]
)
problem.SetVolumetricSources(volumetric_sources=[VolumetricSource(
    block_ids=[0], group_strength=source_strength_high_to_low.tolist()
)])

# --- Solve and write ascending-energy results -----------------------------
solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
solver.Initialize()
solver.Execute()
balance_values = {name: float(value) for name, value in solver.ComputeBalanceTable().items()}
fields_high_to_low = problem.GetScalarFluxFieldFunction()
if len(fields_high_to_low) != NUM_GROUPS:
    raise ValueError("OpenSn did not expose one scalar-flux field per group")


def volume_integral(field, volume):
    """Integrate one group flux over a selected logical volume."""
    interpolation = FieldFunctionInterpolationVolume()
    interpolation.SetOperationType("sum")
    interpolation.SetLogicalVolume(volume)
    interpolation.AddFieldFunction(field)
    interpolation.Execute()
    return float(interpolation.GetValue())


target_high_to_low = np.asarray([
    volume_integral(field, target_volume) for field in fields_high_to_low
])
whole_high_to_low = np.asarray([
    volume_integral(field, whole_volume) for field in fields_high_to_low
])
domain_results = {}
for material in CONFIG["materials"]:
    if material["role"] in ("homogeneous", "target"):
        flux_ascending = target_high_to_low[::-1]
    else:
        flux_ascending = (whole_high_to_low - target_high_to_low)[::-1]
    domain_results[material["logical_name"]] = {
        "block": material["opensn_block"],
        "volume_cm3": measured_volumes[material["opensn_block"]],
        "flux": flux_ascending.tolist(),
    }
primary = next(
    item["logical_name"] for item in CONFIG["materials"]
    if item["role"] in ("homogeneous", "target")
)
balance = balance_values.get(
    "balance",
    balance_values.get("production_rate", 0.0) + balance_values.get("inflow_rate", 0.0)
    - balance_values.get("absorption_rate", 0.0) - balance_values.get("outflow_rate", 0.0),
)
result = {
    "energy_bounds": ENERGY_BOUNDS_EV_ASCENDING.tolist(),
    "flux": domain_results[primary]["flux"],
    "logical_domain": primary,
    "domains": domain_results,
    "solver": {
        "converged": None,
        "iterations": None,
        "residual": None,
        "tolerance": CONFIG["gmres_tolerance"],
        "maximum_iterations": CONFIG["gmres_max_iterations"],
        "balance": float(balance),
    },
}
if globals().get("rank", 0) == 0:
    # MPI ranks share RESULT_PATH; only rank zero writes the compact handoff.
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\\n"
    )
'''
    replacements = {
        "__CASE_NAME__": repr(case.name),
        "__MATERIALS__": pprint.pformat(materials, width=100, sort_dicts=False),
        "__GEOMETRY__": pprint.pformat(geometry, width=100, sort_dicts=False),
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


def _convergence(output: str, tolerance: float, maximum_iterations: int):
    # OpenSn/PETSc versions emit two known residual formats.  A small residual
    # alone is insufficient: the log must also state explicit convergence.
    matches = _ITERATION_RESIDUAL.findall(output) or _PETSC_RESIDUAL.findall(output)
    if not matches:
        raise RuntimeError("OpenSn process succeeded but convergence is unknown")
    iterations, residual = int(matches[-1][0]), float(matches[-1][1])
    explicitly_failed = bool(re.search(r"\b(?:diverged|not\s+converged|converged\s*[:=]\s*false)\b", output, re.I))
    explicitly_converged = bool(re.search(r"status\s*[:=]\s*converged\b", output, re.I))
    converged = (
        explicitly_converged and not explicitly_failed
        and residual <= tolerance and iterations <= maximum_iterations
    )
    if not converged:
        raise RuntimeError(
            f"OpenSn did not converge: iteration={iterations}, residual={residual:.8g}, "
            f"tolerance={tolerance:.8g}"
        )
    return iterations, residual


def run_opensn(
    run_directory,
    *,
    executable,
    mpi_executable=None,
    ranks=1,
    timeout=None,
) -> OpenSnResult:
    """Run generated OpenSn input, preserving logs and requiring convergence."""
    run = Path(run_directory).resolve()
    input_path = run / "opensn" / "input.py"
    mgxs = run / "openmc" / "mgxs.h5"
    if not input_path.is_file():
        raise FileNotFoundError(f"generated OpenSn input is missing: {input_path}")
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
        command = [_executable(mpi_executable), "-n", str(ranks), console, "-i", str(input_path)]
    identity = subprocess.run(
        [console, "--help"], capture_output=True, text=True, timeout=timeout
    )
    version = re.search(r"OpenSn version\s+([^\s]+)", identity.stdout + identity.stderr)
    # OpenSn 1.0.1 prints valid identity/help text but exits nonzero for --help.
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
    output = stdout_path.read_text() + stderr_path.read_text()
    result_path = run / "opensn" / "opensn_result.json"
    if not result_path.is_file():
        raise FileNotFoundError("OpenSn completed without writing opensn_result.json")
    try:
        document = json.loads(result_path.read_text())
        solver = document["solver"]
        tolerance = float(solver["tolerance"])
        maximum = int(solver["maximum_iterations"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("OpenSn result is malformed") from error
    iterations, residual = _convergence(output, tolerance, maximum)
    solver.update(converged=True, iterations=iterations, residual=residual)
    result_path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
    iterations = solver.get("iterations", solver.get("iteration_count"))
    residual = solver.get("residual", solver.get("final_residual"))
    if converged is not True or iterations is None or residual is None:
        raise RuntimeError("OpenSn result does not explicitly report convergence")
    bounds = np.asarray(document["energy_bounds"], dtype=float)
    primary = Spectrum(
        bounds,
        np.asarray(document["flux"], dtype=float),
        logical_domain=document.get("logical_domain"),
    )
    domains = None
    if "domains" in document:
        domains = {
            name: Spectrum(bounds, np.asarray(record["flux"], dtype=float), logical_domain=name)
            for name, record in document["domains"].items()
        }
    return OpenSnResult(
        primary,
        True,
        int(iterations),
        float(residual),
        float(solver["balance"]) if solver.get("balance") is not None else None,
        domains,
    )
