from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from generate_mgxs import Case, Material


OPENMC_PYTHON = Path("/home/ragusa/miniforge3/envs/openmc-env/bin/python")
OPENMC_DATA = Path("/home/ragusa/xs/endfb-viii.0-hdf5/cross_sections.xml")
OPENSN = Path("/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-console")
OPENSN_MPI = Path("/home/ragusa/opt/opensn/commit-b39f7be8a215/bin/opensn-mpiexec")
OPENSN_FISSION_MGXS = Path("/home/ragusa/repo/opensn/test/assets/xs/u235_84g.h5")
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
    outer = None
    target = (1.0, 1.0, 1.0)
    if two_material:
        materials = (
            material("moderator", "moderator"),
            material("target", "target"),
        )
        outer = (1.0, 1.0, 1.0)
        target = (0.4, 0.4, 0.4)
    return Case(
        name="tiny_two" if two_material else "tiny_one",
        materials=materials,
        energy_groups=(1.0e-5, 1.0e6, 2.0e7),
        source_probabilities=(0.25, 0.75),
        source_kind="grouped",
        target_dimensions_cm=target,
        outer_dimensions_cm=outer,
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
