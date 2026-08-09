"""Generate, execute, and read explicit OpenMC calculations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pprint
import shutil
import subprocess
import sys

import numpy as np

from .case import Case, _artifact, _material_records, _update_run_metadata
from .results import Spectrum


def _write_openmc_input(case: Case, path: Path) -> None:
    materials = _material_records(case)
    geometry = {
        "type": case.geometry_type,
        "target_dimensions_cm": case.target_dimensions_cm,
        "outer_dimensions_cm": case.outer_dimensions_cm or case.target_dimensions_cm,
        "boundaries": dict(zip(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), case.boundaries)),
    }
    history = {
        "particles_per_batch": case.particles_per_batch,
        "batches": case.batches,
        "total_histories": case.total_histories,
    }
    text = '''\
"""Generated fixed-source OpenMC model and MGXS processing entry point.

Run from this directory with ``python model.py run`` and later
``python model.py process``.  All paths below are relative to the prepared run.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs


# --- Human-readable case definition ---------------------------------------
CASE_NAME = __CASE_NAME__
MATERIALS = __MATERIALS__

# Geometry dimensions are full lengths in cm.  Material roles, rather than
# caller ordering, determine target and moderator regions.
GEOMETRY = __GEOMETRY__

# This physical source is the sole authority for both continuous-energy OpenMC
# sampling and the group probabilities derived by the OpenSn input.
PHYSICAL_SOURCE = __PHYSICAL_SOURCE__

# OpenMC interprets particles as particles per batch, not total histories.
OPENMC_HISTORY_SETTINGS = __HISTORY__

# The Legendre order is shared by tally production and downstream transport.
MGXS_SETTINGS = __MGXS_SETTINGS__


# --- Energy-group data and run-relative paths -----------------------------
# Python-facing boundaries are always low-to-high in physical energy.
ENERGY_BOUNDS_EV = np.asarray(__ENERGY_BOUNDS__, dtype=float)
RUN_DIRECTORY = Path(__file__).resolve().parents[1]
OPENMC_DIRECTORY = RUN_DIRECTORY / "openmc"
MGXS_TYPES = [
    "total", "absorption", "capture", "fission", "nu-fission",
    "consistent scatter matrix", "consistent nu-scatter matrix", "chi", "chi-prompt",
]

CONFIG = {
    "case_name": CASE_NAME,
    "materials": MATERIALS,
    "geometry_type": GEOMETRY["type"],
    "target_dimensions_cm": GEOMETRY["target_dimensions_cm"],
    "outer_dimensions_cm": GEOMETRY["outer_dimensions_cm"],
    "boundaries": GEOMETRY["boundaries"],
    "batches": OPENMC_HISTORY_SETTINGS["batches"],
    "particles_per_batch": OPENMC_HISTORY_SETTINGS["particles_per_batch"],
    "scattering_order": MGXS_SETTINGS["scattering_order"],
}


def source_probabilities():
    """Derive ascending group masses from the generated physical source."""
    kind = PHYSICAL_SOURCE["kind"]
    if kind == "grouped":
        masses = np.asarray(PHYSICAL_SOURCE["probabilities"], dtype=float)
        if masses.shape != (ENERGY_BOUNDS_EV.size - 1,) or np.any(masses < 0.0):
            raise ValueError("source must provide one nonnegative mass per group")
        if not np.isclose(masses.sum(), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("explicit group probabilities must sum to one")
        return masses
    elif kind == "uniform_energy":
        masses = np.diff(ENERGY_BOUNDS_EV)
    elif kind == "watt":
        bounds_mev = ENERGY_BOUNDS_EV / 1.0e6
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
    if masses.shape != (ENERGY_BOUNDS_EV.size - 1,) or np.any(masses < 0.0):
        raise ValueError("source must provide one nonnegative mass per group")
    masses = masses / masses.sum()
    masses[-1] += 1.0 - masses.sum()
    return masses


SOURCE_PROBABILITIES_ASCENDING = source_probabilities()


# --- Materials, geometry, source, and tallies -----------------------------
def build_model():
    """Construct the complete native OpenMC model and MGXS library."""
    material_by_domain = {}
    for definition in CONFIG["materials"]:
        material = openmc.Material(definition["openmc_id"], definition["name"])
        material.set_density("g/cm3", definition["density_g_cm3"])
        material.temperature = definition["temperature_k"]
        for isotope, atom_fraction in definition["isotopes"]:
            material.add_nuclide(isotope, atom_fraction, percent_type="ao")
        for table in definition["thermal_scattering"]:
            material.add_s_alpha_beta(table)
        material_by_domain[definition["logical_name"]] = material

    model = openmc.Model()
    model.materials = openmc.Materials(material_by_domain.values())
    if os.getenv("OPENMC_CROSS_SECTIONS"):
        model.materials.cross_sections = os.environ["OPENMC_CROSS_SECTIONS"]

    target = next(item for item in CONFIG["materials"] if item["role"] in ("homogeneous", "target"))
    target_material = material_by_domain[target["logical_name"]]
    tdx, tdy, tdz = CONFIG["target_dimensions_cm"]
    if CONFIG["geometry_type"] == "homogeneous":
        surfaces = (
            openmc.XPlane(x0=-tdx / 2, boundary_type=CONFIG["boundaries"]["xmin"]),
            openmc.XPlane(x0=+tdx / 2, boundary_type=CONFIG["boundaries"]["xmax"]),
            openmc.YPlane(y0=-tdy / 2, boundary_type=CONFIG["boundaries"]["ymin"]),
            openmc.YPlane(y0=+tdy / 2, boundary_type=CONFIG["boundaries"]["ymax"]),
            openmc.ZPlane(z0=-tdz / 2, boundary_type=CONFIG["boundaries"]["zmin"]),
            openmc.ZPlane(z0=+tdz / 2, boundary_type=CONFIG["boundaries"]["zmax"]),
        )
        xmin, xmax, ymin, ymax, zmin, zmax = surfaces
        target_region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax
        cells = [openmc.Cell(fill=target_material, region=target_region)]
    else:
        tx0, tx1 = openmc.XPlane(-tdx / 2), openmc.XPlane(+tdx / 2)
        ty0, ty1 = openmc.YPlane(-tdy / 2), openmc.YPlane(+tdy / 2)
        tz0, tz1 = openmc.ZPlane(-tdz / 2), openmc.ZPlane(+tdz / 2)
        target_region = +tx0 & -tx1 & +ty0 & -ty1 & +tz0 & -tz1
        odx, ody, odz = CONFIG["outer_dimensions_cm"]
        ox0 = openmc.XPlane(-odx / 2, boundary_type=CONFIG["boundaries"]["xmin"])
        ox1 = openmc.XPlane(+odx / 2, boundary_type=CONFIG["boundaries"]["xmax"])
        oy0 = openmc.YPlane(-ody / 2, boundary_type=CONFIG["boundaries"]["ymin"])
        oy1 = openmc.YPlane(+ody / 2, boundary_type=CONFIG["boundaries"]["ymax"])
        oz0 = openmc.ZPlane(-odz / 2, boundary_type=CONFIG["boundaries"]["zmin"])
        oz1 = openmc.ZPlane(+odz / 2, boundary_type=CONFIG["boundaries"]["zmax"])
        outer_region = +ox0 & -ox1 & +oy0 & -oy1 & +oz0 & -oz1
        moderator = next(item for item in CONFIG["materials"] if item["role"] == "moderator")
        cells = [
            openmc.Cell(fill=target_material, region=target_region),
            openmc.Cell(fill=material_by_domain[moderator["logical_name"]], region=outer_region & ~target_region),
        ]
    model.geometry = openmc.Geometry(cells)

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = CONFIG["batches"]
    settings.particles = CONFIG["particles_per_batch"]
    settings.temperature = {"default": CONFIG["materials"][0]["temperature_k"]}
    source = openmc.IndependentSource()
    source.space = openmc.stats.Box(
        (-tdx / 2, -tdy / 2, -tdz / 2), (tdx / 2, tdy / 2, tdz / 2),
        only_fissionable=False,
    )
    source.angle = openmc.stats.Isotropic()
    if PHYSICAL_SOURCE["kind"] == "uniform_energy":
        source.energy = openmc.stats.Uniform(ENERGY_BOUNDS_EV[0], ENERGY_BOUNDS_EV[-1])
    elif PHYSICAL_SOURCE["kind"] == "watt":
        source.energy = openmc.stats.Watt(
            a=PHYSICAL_SOURCE["a_mev"] * 1.0e6,
            b=PHYSICAL_SOURCE["b_per_mev"] / 1.0e6,
        )
        source.constraints = {
            "energy_bounds": (ENERGY_BOUNDS_EV[0], ENERGY_BOUNDS_EV[-1]),
            "rejection_strategy": "resample",
        }
    else:
        densities = SOURCE_PROBABILITIES_ASCENDING / np.diff(ENERGY_BOUNDS_EV)
        source.energy = openmc.stats.Tabular(
            ENERGY_BOUNDS_EV,
            np.append(densities, densities[-1]),
            interpolation="histogram",
        )
    source.particle = "neutron"
    settings.source = [source]
    model.settings = settings

    groups = openmc.mgxs.EnergyGroups(group_edges=ENERGY_BOUNDS_EV)
    library = openmc.mgxs.Library(model.geometry)
    library.energy_groups = groups
    library.domain_type = "material"
    library.domains = [material_by_domain[item["logical_name"]] for item in CONFIG["materials"]]
    library.mgxs_types = MGXS_TYPES
    library.by_nuclide = False
    # The direct and OpenSn equations use the physical total, not P0-corrected transport.
    library.correction = None
    library.legendre_order = CONFIG["scattering_order"]
    library.check_library_for_openmc_mgxs()
    library.build_library()
    tallies = openmc.Tallies()
    library.add_to_tallies_file(tallies, merge=True)
    model.tallies = tallies
    return model, library


MODEL, MGXS_LIBRARY = build_model()


def write_inputs():
    """Write model.xml for inspection without running particle transport."""
    MODEL.export_to_model_xml()


def run_transport():
    """Run OpenMC and write its statepoint in this openmc directory."""
    MODEL.run(threads=int(os.getenv("MGXS_PROCESSES", "1")))


def _canonical_xs(mgxs, quantity):
    # OpenMC APIs can squeeze P0 and expose solver-native group axes.  Preserve
    # uncertainty in explicit arrays matching the exported MGXS quantities.
    if quantity == "scatter":
        values = mgxs.get_xs(
            order_groups="decreasing", row_column="inout", value="std_dev"
        )
        values = np.asarray(values, dtype=float)
        if values.ndim == 2:  # OpenMC squeezes the sole P0 moment.
            values = values[..., np.newaxis]
        return np.moveaxis(values, -1, 0)
    return np.asarray(
        mgxs.get_xs(order_groups="decreasing", value="std_dev"), dtype=float
    ).reshape(-1)


def process_statepoint(statepoint=None):
    """Create mgxs.h5, uncertainty diagnostics, and the compact flux result."""
    statepoint = Path(statepoint or OPENMC_DIRECTORY / f"statepoint.{CONFIG['batches']}.h5")
    if not statepoint.is_file():
        raise FileNotFoundError(f"expected OpenMC statepoint is missing: {statepoint}")
    with openmc.StatePoint(statepoint) as statepoint_file:
        MGXS_LIBRARY.load_from_statepoint(statepoint_file)

    names = [item["logical_name"] for item in CONFIG["materials"]]
    # mgxs.h5 is the only cross-section handoff consumed by generated OpenSn.
    mean_library = MGXS_LIBRARY.create_mg_library(xsdata_names=names)
    mean_library.export_to_hdf5(str(OPENMC_DIRECTORY / "mgxs.h5"))

    uncertainty = {
        "statepoint": statepoint.name,
        "openmc_version": openmc.__version__,
        "domains": {},
    }
    spectra = {}
    for item, domain in zip(CONFIG["materials"], MGXS_LIBRARY.domains):
        total = MGXS_LIBRARY.get_mgxs(domain, "total")
        flux = np.asarray(total.get_flux(order_groups="decreasing", value="mean"), dtype=float).reshape(-1)
        flux_std = np.asarray(total.get_flux(order_groups="decreasing", value="std_dev"), dtype=float).reshape(-1)
        spectra[item["logical_name"]] = {"flux": flux.tolist(), "std_dev": flux_std.tolist()}
        quantities = {
            "total": _canonical_xs(total, "total"),
            "absorption": _canonical_xs(MGXS_LIBRARY.get_mgxs(domain, "absorption"), "absorption"),
            "scatter": _canonical_xs(MGXS_LIBRARY.get_mgxs(domain, "consistent scatter matrix"), "scatter"),
        }
        nu_fission = MGXS_LIBRARY.get_mgxs(domain, "nu-fission")
        nu_mean = np.asarray(nu_fission.get_xs(order_groups="decreasing"), dtype=float).reshape(-1)
        if np.any(nu_mean > 0.0):
            quantities["fission"] = _canonical_xs(MGXS_LIBRARY.get_mgxs(domain, "fission"), "fission")
            quantities["nu_fission"] = _canonical_xs(nu_fission, "nu_fission")
            chi = MGXS_LIBRARY.get_mgxs(domain, "chi")
            quantities["chi_raw"] = _canonical_xs(chi, "chi_raw")
            raw = np.asarray(chi.get_xs(order_groups="decreasing"), dtype=float).reshape(-1)
            # Raw chi uncertainty cannot be reused after normalization.  Store
            # the normalization audit separately and leave normalized uncertainty unset.
            raw_sum = float(raw.sum())
            factor = 1.0 / raw_sum
            normalized = raw * factor
            index = int(np.argmax(normalized))
            correction = float(1.0 - normalized.sum())
            uncertainty["domains"].setdefault(item["logical_name"], {})["chi"] = {
                "raw_sum": raw_sum,
                "normalization_factor": factor,
                "closure_index": index,
                "closure_correction": correction,
                "normalized_uncertainty": None,
            }
        domain_record = uncertainty["domains"].setdefault(item["logical_name"], {})
        domain_record["temperature_k"] = item["temperature_k"]
        domain_record["quantities"] = {
            name: {"std_dev": values.tolist()} for name, values in quantities.items()
        }

    diagnostics = RUN_DIRECTORY / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "mgxs_uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True, allow_nan=False) + "\\n"
    )
    primary = next(item["logical_name"] for item in CONFIG["materials"] if item["role"] in ("homogeneous", "target"))
    result = {
        "energy_bounds": ENERGY_BOUNDS_EV.tolist(),
        "flux": spectra[primary]["flux"],
        "std_dev": spectra[primary]["std_dev"],
        "logical_domain": primary,
        "domains": spectra,
    }
    (OPENMC_DIRECTORY / "openmc_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\\n"
    )


def main(argv=None):
    """Dispatch the write-input, transport, or statepoint-processing operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("write-input", "run", "process"))
    parser.add_argument("--statepoint", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "write-input":
        write_inputs()
    elif args.operation == "run":
        run_transport()
    else:
        process_statepoint(args.statepoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
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
        "__ENERGY_BOUNDS__": pprint.pformat(
            case.energy_bounds_ev, width=100, sort_dicts=False
        ),
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
    run = Path(run_directory).resolve()
    model = run / "openmc" / "model.py"
    if not model.is_file():
        raise FileNotFoundError(f"generated OpenMC input is missing: {model}")
    python = _executable(python_executable)
    cross_sections = Path(cross_sections).resolve()
    if not cross_sections.is_file():
        raise FileNotFoundError(f"OpenMC cross_sections.xml is missing: {cross_sections}")
    identity = subprocess.run(
        [python, "-c", "import openmc; print(openmc.__version__)"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if identity.returncode or not identity.stdout.strip():
        raise RuntimeError("OpenMC software identity could not be established")
    if operation not in {"write-input", "run", "process", "all"}:
        raise ValueError("operation must be write-input, run, process, or all")
    operations = ("run", "process") if operation == "all" else (operation,)
    commands = [[python, str(model), item] for item in operations]
    openmc_metadata = {
        "version": identity.stdout.strip().splitlines()[-1],
        "python": python,
        "nuclear_data": {
            "cross_sections": str(cross_sections),
            "bytes": cross_sections.stat().st_size,
            "sha256": __import__("hashlib").sha256(cross_sections.read_bytes()).hexdigest(),
        },
        "commands": commands,
    }
    _update_run_metadata(run, openmc=openmc_metadata)
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = str(Path(python).parent) + os.pathsep + environment.get("PATH", "")
    environment["OPENMC_CROSS_SECTIONS"] = str(cross_sections)
    environment["MGXS_PROCESSES"] = str(threads)
    with (logs / "openmc.stdout").open("w") as stdout, (logs / "openmc.stderr").open("w") as stderr:
        for command in commands:
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


def load_openmc_result(path) -> Spectrum:
    """Load the canonical compact OpenMC spectrum result."""
    path = Path(path)
    if path.is_dir():
        path = path / "openmc" / "openmc_result.json"
    document = json.loads(path.read_text())
    if "std_dev" not in document:
        raise ValueError("OpenMC result does not contain statistical uncertainty")
    return Spectrum(
        np.asarray(document["energy_bounds"], dtype=float),
        np.asarray(document["flux"], dtype=float),
        np.asarray(document["std_dev"], dtype=float),
        document.get("logical_domain"),
    )
