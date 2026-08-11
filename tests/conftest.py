from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from generate_mgxs import Case, Material, NestedBoxGeometry


def _runtime_path(environment_variable: str, *, default=None) -> Path:
    """Return an optional test runtime path without embedding a local layout."""
    value = os.environ.get(environment_variable, default)
    if value:
        return Path(value).expanduser()
    return Path("__generate_mgxs_runtime_not_configured__")


OPENMC_PYTHON = _runtime_path("OPENMC_PYTHON", default=sys.executable)
OPENMC_DATA = _runtime_path("OPENMC_CROSS_SECTIONS")
OPENSN = _runtime_path("OPENSN_CONSOLE")
OPENSN_MPI = _runtime_path("OPENSN_MPIEXEC")
OPENSN_FISSION_MGXS = _runtime_path("OPENSN_FISSION_MGXS")
EVIDENCE = Path(__file__).parent / "data"


def material(name="one", role="homogeneous"):
    """Return a minimal material; solver IDs are intentionally not user input."""
    return Material(
        logical_name=name,
        name=name,
        density_g_cm3=1.0,
        composition=(("H1", 1.0),),
        role=role,
    )


def tiny_case(two_material=False, *, max_iterations=50):
    materials = (material(),)
    target = (1.0, 1.0, 1.0)
    geometry = None
    if two_material:
        materials = (
            material("moderator", "moderator"),
            material("target", "target"),
        )
        target = (0.4, 0.4, 0.4)
        geometry = NestedBoxGeometry(
            target=materials[1],
            moderator=materials[0],
            target_dimensions_cm=target,
            outer_dimensions_cm=(1.0, 1.0, 1.0),
        )
    return Case(
        name="tiny_two" if two_material else "tiny_one",
        materials=materials,
        energy_groups=(1.0e-5, 1.0e6, 2.0e7),
        source_probabilities=(0.25, 0.75),
        source_kind="grouped",
        target_dimensions_cm=target if geometry is None else None,
        geometry=geometry,
        scattering_order=0,
        gmres_tolerance=1.0e-8,
        gmres_max_iterations=max_iterations,
        gmres_restart=20,
        particles_per_batch=10,
        batches=2,
    )


def write_tiny_mgxs(path: Path, domains=("one",)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["filetype"] = np.bytes_("mgxs")
        h5.attrs["version"] = [1, 0]
        h5.attrs["energy_groups"] = 2
        h5.attrs["delayed_groups"] = 0
        h5.attrs["group structure"] = [1.0e-5, 1.0e6, 2.0e7]
        for name in domains:
            domain = h5.create_group(name)
            domain.attrs["fissionable"] = False
            domain.attrs["order"] = 0
            domain.attrs["representation"] = np.bytes_("isotropic")
            domain.attrs["scatter_format"] = np.bytes_("legendre")
            domain.attrs["scatter_shape"] = np.bytes_("[G][G'][Order]")
            domain.create_group("kTs").create_dataset("294K", data=294.0 * 8.617333262145e-5)
            temperature = domain.create_group("294K")
            temperature.create_dataset("total", data=[1.0, 1.0])
            temperature.create_dataset("absorption", data=[0.9, 0.9])
            scatter = temperature.create_group("scatter_data")
            scatter.create_dataset("g_min", data=[1, 2])
            scatter.create_dataset("g_max", data=[1, 2])
            scatter.create_dataset("scatter_matrix", data=[0.1, 0.1])
            scatter.create_dataset("multiplicity_matrix", data=[1.0, 1.0])


def write_result(path: Path, *, converged=None, malformed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if malformed:
        path.write_text("not json")
        return
    document = {
        "energy_bounds": [1.0e-5, 1.0e6, 2.0e7],
        "flux": [1.0, 2.0],
        "solver": {
            "converged": converged,
            "iterations": None,
            "residual": None,
            "tolerance": 1.0e-8,
            "maximum_iterations": 50,
            "balance": 0.0,
        },
    }
    path.write_text(json.dumps(document))


@pytest.fixture
def one_case():
    return tiny_case()


@pytest.fixture
def two_case():
    return tiny_case(True)
