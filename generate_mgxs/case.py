"""Scientific case definitions and run-directory preparation."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Literal

import numpy as np


# These two seed-specific structures are not present in OpenMC 0.15's standard
# group library. Standard structures are resolved lazily from OpenMC instead of
# copying OpenMC-owned boundary tables into this package.
_WIMS69_MEV = (
    1e-11, 5e-9, 1e-8, 1.5e-8, 2e-8, 2.5e-8, 3e-8, 3.5e-8, 4.2e-8,
    5e-8, 5.8e-8, 6.7e-8, 8e-8, 1e-7, 1.4e-7, 1.8e-7, 2.2e-7,
    2.5e-7, 2.8e-7, 3e-7, 3.2e-7, 3.5e-7, 4e-7, 5e-7, 6.25e-7,
    7.8e-7, 8.5e-7, 9.1e-7, 9.5e-7, 9.72e-7, 9.96e-7, 1.02e-6,
    1.045e-6, 1.071e-6, 1.097e-6, 1.123e-6, 1.15e-6, 1.3e-6,
    1.5e-6, 2.1e-6, 2.6e-6, 3.3e-6, 4e-6, 9.877e-6, 1.5968e-5,
    2.77e-5, 4.8052e-5, 7.55014e-5, 1.48729e-4, 3.67263e-4,
    9.06899e-4, 1.4251e-3, 2.23945e-3, 3.5191e-3, 5.53e-3,
    9.118e-3, 1.503e-2, 2.478e-2, 4.085e-2, 6.734e-2, 1.11e-1,
    1.83e-1, 3.025e-1, 5e-1, 8.21e-1, 1.353, 2.231, 3.679, 6.0655,
    10.0,
)
_LANL30_MEV = (
    1.39e-10, 1.52e-7, 4.14e-7, 1.13e-6, 3.06e-6, 8.32e-6, 2.26e-5,
    6.14e-5, 1.67e-4, 4.54e-4, 1.235e-3, 3.35e-3, 9.12e-3, 2.48e-2,
    6.76e-2, 0.184, 0.303, 0.5, 0.823, 1.353, 1.738, 2.232, 2.865,
    3.68, 6.07, 7.79, 10.0, 12.0, 13.5, 15.0, 17.0,
)

# This is deliberately a naming rule, not a periodic-table database. OpenMC is
# the scientific authority that expands natural elements and validates whether
# a canonical element or nuclide is available in the configured nuclear data.
_COMPOSITION_IDENTIFIER = re.compile(
    r"^[A-Z][a-z]?(?:[1-9]\d*(?:_m[1-9]\d*)?)?$"
)


def _ascending(values, *, expected: int | None = None, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if expected is not None and array.size != expected:
        raise ValueError(f"{name} has the wrong shape")
    if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly ascending")
    return array


def _positive_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer(value, name: str, *, minimum: int) -> int:
    # bool is an int subclass, but accepting True as one particle or one angle
    # silently masks malformed scientific input.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _openmc_energy_bounds(name: str) -> tuple[float, ...]:
    """Resolve one canonical OpenMC group name without a module-level import."""
    try:
        from openmc.mgxs import GROUP_STRUCTURES
    except ImportError as error:
        raise ImportError(
            "OpenMC is required to resolve a named energy-group structure"
        ) from error

    try:
        values = GROUP_STRUCTURES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown OpenMC energy-group structure {name!r}"
        ) from error

    return tuple(
        float(value)
        for value in _ascending(values, name=f"OpenMC group structure {name!r}")
    )


def energy_bounds(name: str) -> tuple[float, ...]:
    """Return custom or OpenMC-standard ascending energy boundaries in eV."""
    if not isinstance(name, str) or not name:
        raise ValueError("energy-group structure name must be a nonempty string")

    custom = {
        "WIMS69": _WIMS69_MEV,
        "LANL30": _LANL30_MEV,
    }
    try:
        # If OpenMC acquires a structure with the same spelling as a legacy
        # custom name, OpenMC becomes the authority automatically.
        return _openmc_energy_bounds(name)
    except (ImportError, ValueError):
        if name in custom:
            return tuple(float(value * 1.0e6) for value in custom[name])
        raise


def source_probabilities(
    bounds_ev, kind: Literal["uniform_energy", "watt"], *, a_mev=0.988, b_per_mev=2.249
) -> tuple[float, ...]:
    """Integrate a physical source into ascending-energy group probabilities."""
    bounds = _ascending(bounds_ev, name="energy boundaries")

    if kind == "uniform_energy":
        masses = np.diff(bounds)
    elif kind == "watt":
        a_mev = _positive_float(a_mev, "Watt a_mev")
        b_per_mev = _positive_float(b_per_mev, "Watt b_per_mev")
        bounds_mev = bounds / 1.0e6

        # Fixed-order Gauss-Legendre integration makes generated cases
        # deterministic while accurately resolving the smooth Watt spectrum.
        nodes, weights = np.polynomial.legendre.leggauss(64)
        masses = []
        for low, high in zip(bounds_mev[:-1], bounds_mev[1:]):
            energies = 0.5 * (high - low) * nodes + 0.5 * (high + low)
            density = np.exp(-energies / a_mev) * np.sinh(
                np.sqrt(b_per_mev * energies)
            )
            masses.append(0.5 * (high - low) * np.dot(weights, density))
        masses = np.asarray(masses)
    else:
        raise ValueError("source kind must be 'uniform_energy' or 'watt'")

    if (
        not np.all(np.isfinite(masses))
        or np.any(masses < 0.0)
        or masses.sum() <= 0.0
    ):
        raise ValueError("source integration produced invalid group masses")

    masses /= masses.sum()

    # Close the probability vector exactly enough for downstream serialization;
    # the correction is roundoff-sized and is applied to a populated end group.
    masses[-1] += 1.0 - masses.sum()

    return tuple(float(value) for value in masses)


class Material:
    """A material with nonnegative relative atomic composition amounts."""

    def __init__(
        self,
        logical_name: str,
        name: str,
        density_g_cm3: float,
        composition,
        temperature_k: float = 294.0,
        thermal_scattering=(),
        role: Literal["homogeneous", "target", "moderator"] = "homogeneous",
    ):
        if not isinstance(logical_name, str) or not logical_name.strip():
            raise ValueError("logical_name cannot be empty")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("material name cannot be empty")
        if role not in {"homogeneous", "target", "moderator"}:
            raise ValueError("material role must be homogeneous, target, or moderator")

        try:
            components = tuple(
                (str(identifier), float(amount))
                for identifier, amount in composition
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "material composition must contain identifier/amount pairs"
            ) from error

        if not components:
            raise ValueError("material composition cannot be empty")
        invalid = [
            identifier
            for identifier, _ in components
            if not _COMPOSITION_IDENTIFIER.fullmatch(identifier)
        ]
        if invalid:
            raise ValueError(
                f"invalid material composition identifier {invalid[0]!r}"
            )

        amounts = np.asarray([amount for _, amount in components], dtype=float)
        if (
            not np.all(np.isfinite(amounts))
            or np.any(amounts < 0.0)
            or not np.any(amounts > 0.0)
        ):
            raise ValueError(
                "material atom amounts must be finite, nonnegative, and not all zero"
            )

        thermal_scattering = tuple(str(table) for table in thermal_scattering)
        if any(not table.strip() for table in thermal_scattering):
            raise ValueError("thermal-scattering table names cannot be empty")

        self.logical_name = logical_name
        self.name = name
        self.density_g_cm3 = _positive_float(density_g_cm3, "material density")
        self.composition = components
        self.temperature_k = _positive_float(temperature_k, "material temperature")
        self.thermal_scattering = thermal_scattering
        self.role = role


class Case:
    """One independently preparable fixed-source or homogeneous eigenvalue case."""

    def __init__(
        self,
        *,
        name: str,
        materials,
        energy_groups,
        target_dimensions_cm,
        source_kind: Literal["uniform_energy", "watt", "grouped"] | None = None,
        run_mode: Literal["fixed_source", "eigenvalue"] = "fixed_source",
        source_probabilities=None,
        outer_dimensions_cm=None,
        boundaries=("reflective",) * 6,
        particles_per_batch: int = 25_000,
        batches: int = 40,
        inactive_batches: int = 0,
        scattering_order: int = 3,
        gmres_tolerance: float = 1.0e-10,
        gmres_max_iterations: int = 1200,
        gmres_restart: int = 100,
        keigen_tolerance: float = 1.0e-8,
        keigen_max_iterations: int = 1000,
        watt_a_mev: float = 0.988,
        watt_b_per_mev: float = 2.249,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("case name cannot be empty")
        materials = tuple(materials)
        if (
            not materials
            or len(materials) > 2
            or not all(isinstance(material, Material) for material in materials)
        ):
            raise ValueError("a case requires one material or a target and moderator")

        logical_names = [material.logical_name for material in materials]
        if len(set(logical_names)) != len(logical_names):
            raise ValueError("logical material names must be unique")

        roles = {material.role for material in materials}
        if len(materials) == 1 and roles != {"homogeneous"}:
            raise ValueError("one material must have the homogeneous role")
        if len(materials) == 2 and roles != {"target", "moderator"}:
            raise ValueError("two materials must have target and moderator roles")

        if isinstance(energy_groups, str):
            energy_group_structure = energy_groups
            bounds = _openmc_energy_bounds(energy_group_structure)
        else:
            energy_group_structure = None
            bounds = tuple(
                float(value)
                for value in _ascending(energy_groups, name="energy boundaries")
            )

        if run_mode not in {"fixed_source", "eigenvalue"}:
            raise ValueError("run_mode must be fixed_source or eigenvalue")
        if run_mode == "eigenvalue" and len(materials) != 1:
            raise ValueError("eigenvalue cases require one homogeneous material")
        if run_mode == "fixed_source":
            if source_kind not in {"uniform_energy", "watt", "grouped"}:
                raise ValueError(
                    "fixed-source cases require source kind to be "
                    "uniform_energy, watt, or grouped"
                )
        elif source_kind is not None:
            raise ValueError("eigenvalue cases do not accept source_kind")

        if run_mode == "fixed_source":
            watt_a_mev = _positive_float(watt_a_mev, "Watt a_mev")
            watt_b_per_mev = _positive_float(watt_b_per_mev, "Watt b_per_mev")
        else:
            watt_a_mev = None
            watt_b_per_mev = None
        grouped = None

        # Only a grouped source owns an explicit probability vector. Uniform
        # and Watt vectors are derived later from their physical definitions.
        if run_mode == "eigenvalue":
            if source_probabilities is not None:
                raise ValueError("eigenvalue cases do not accept source_probabilities")
        elif source_kind == "grouped":
            if source_probabilities is None:
                raise ValueError("grouped source requires source_probabilities")
            grouped_array = np.asarray(source_probabilities, dtype=float)
            if grouped_array.shape != (len(bounds) - 1,) or not np.all(np.isfinite(grouped_array)):
                raise ValueError("source probabilities must contain one finite value per group")
            if np.any(grouped_array < 0.0) or not np.isclose(
                grouped_array.sum(), 1.0, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError("source probabilities must be nonnegative and sum to one")
            grouped = tuple(float(value) for value in grouped_array)
        elif source_probabilities is not None:
            raise ValueError("only a grouped source accepts source_probabilities")

        target = self._dimensions(target_dimensions_cm, "target dimensions")
        outer = None if outer_dimensions_cm is None else self._dimensions(
            outer_dimensions_cm, "outer dimensions"
        )

        if len(materials) == 2:
            if outer is None or any(outside <= inside for inside, outside in zip(target, outer)):
                raise ValueError("moderated outer dimensions must strictly exceed target dimensions")
        elif outer is not None and outer != target:
            raise ValueError("a homogeneous case does not have distinct outer dimensions")

        boundaries = tuple(boundaries)
        if len(boundaries) != 6 or any(x not in {"reflective", "vacuum"} for x in boundaries):
            raise ValueError("six reflective/vacuum boundary values are required")

        self.name = name
        self.materials = materials
        self.energy_group_structure = energy_group_structure
        self.energy_bounds_ev = bounds
        self.run_mode = run_mode
        self.source_kind = source_kind
        self._grouped_source_probabilities = grouped
        self.target_dimensions_cm = target
        self.outer_dimensions_cm = outer
        self.boundaries = boundaries
        self.particles_per_batch = _integer(
            particles_per_batch, "particles_per_batch", minimum=1
        )
        self.batches = _integer(batches, "batches", minimum=1)
        self.inactive_batches = _integer(
            inactive_batches, "inactive_batches", minimum=0
        )
        if self.run_mode == "fixed_source" and self.inactive_batches != 0:
            raise ValueError("fixed-source cases require inactive_batches=0")
        if (
            self.run_mode == "eigenvalue"
            and self.inactive_batches >= self.batches
        ):
            raise ValueError("eigenvalue inactive_batches must be less than batches")
        self.scattering_order = _integer(scattering_order, "scattering_order", minimum=0)
        self.gmres_tolerance = _positive_float(gmres_tolerance, "gmres_tolerance")
        self.gmres_max_iterations = _integer(
            gmres_max_iterations, "gmres_max_iterations", minimum=1
        )
        self.gmres_restart = _integer(gmres_restart, "gmres_restart", minimum=1)
        self.keigen_tolerance = _positive_float(
            keigen_tolerance, "keigen_tolerance"
        )
        self.keigen_max_iterations = _integer(
            keigen_max_iterations, "keigen_max_iterations", minimum=1
        )

        self.watt_a_mev = watt_a_mev
        self.watt_b_per_mev = watt_b_per_mev

    @staticmethod
    def _dimensions(values, name: str) -> tuple[float, float, float]:
        try:
            dimensions = tuple(float(value) for value in values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain three finite positive values") from error

        if len(dimensions) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in dimensions
        ):
            raise ValueError(f"{name} must contain three finite positive values")
        return dimensions

    @property
    def source_probabilities(self) -> tuple[float, ...]:
        """Return fixed-source group probabilities in ascending energy order."""
        if self.run_mode != "fixed_source":
            raise ValueError("eigenvalue cases have no external source probabilities")
        if self.source_kind == "grouped":
            return self._grouped_source_probabilities
        return source_probabilities(
            self.energy_bounds_ev,
            self.source_kind,
            a_mev=self.watt_a_mev,
            b_per_mev=self.watt_b_per_mev,
        )

    @property
    def source_definition(self) -> dict | None:
        """Return the single physical source definition used by both solvers."""
        if self.run_mode != "fixed_source":
            return None
        if self.source_kind == "uniform_energy":
            return {"kind": "uniform_energy"}
        if self.source_kind == "watt":
            return {
                "kind": "watt",
                "a_mev": self.watt_a_mev,
                "b_per_mev": self.watt_b_per_mev,
            }
        return {"kind": "grouped", "probabilities": self.source_probabilities}

    @property
    def source_volume_cm3(self) -> float:
        """Return the target volume over which the unit source is distributed."""
        return math.prod(self.target_dimensions_cm)

    @property
    def geometry_type(self) -> str:
        """Return the generated geometry implementation selected by material roles."""
        return "homogeneous" if len(self.materials) == 1 else "moderated_target"

    @property
    def total_histories(self) -> int:
        """Return the total source histories requested across all batches."""
        return self.batches * self.particles_per_batch


def _material_records(case: Case) -> list[dict]:
    # OpenMC IDs are serialization details, not physics. Sorting logical names
    # makes them stable even when target/moderator input order changes.
    openmc_ids = {
        name: index for index, name in enumerate(
            sorted(material.logical_name for material in case.materials), start=1
        )
    }
    return [
        {
            "logical_name": material.logical_name,
            "name": material.name,
            "density_g_cm3": material.density_g_cm3,
            "composition": tuple(
                {
                    "identifier": identifier,
                    "kind": (
                        "nuclide"
                        if any(char.isdigit() for char in identifier)
                        else "element"
                    ),
                    "atom_amount": atom_amount,
                }
                for identifier, atom_amount in material.composition
            ),
            "temperature_k": material.temperature_k,
            "thermal_scattering": material.thermal_scattering,
            "role": material.role,
            "openmc_id": openmc_ids[material.logical_name],
        }
        for material in case.materials
    ]


def _jsonable_case(case: Case) -> dict:
    record = {
        "name": case.name,
        "materials": _material_records(case),
        "energy_group_structure": case.energy_group_structure,
        "energy_bounds_ev": case.energy_bounds_ev,
        "run_mode": case.run_mode,
        "target_dimensions_cm": case.target_dimensions_cm,
        "outer_dimensions_cm": case.outer_dimensions_cm,
        "boundaries": case.boundaries,
        "particles_per_batch": case.particles_per_batch,
        "batches": case.batches,
        "inactive_batches": case.inactive_batches,
        "total_histories": case.total_histories,
        "scattering_order": case.scattering_order,
        "gmres_tolerance": case.gmres_tolerance,
        "gmres_max_iterations": case.gmres_max_iterations,
        "gmres_restart": case.gmres_restart,
        "keigen_tolerance": case.keigen_tolerance,
        "keigen_max_iterations": case.keigen_max_iterations,
        "geometry_type": case.geometry_type,
    }
    if case.run_mode == "fixed_source":
        record["source"] = case.source_definition
        record["source_probabilities"] = case.source_probabilities
        record["source_volume_cm3"] = case.source_volume_cm3
    return record


def _artifact(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _update_run_metadata(run_directory: Path, **updates) -> None:
    path = run_directory / "_metadata" / "run.json"
    data = json.loads(path.read_text()) if path.exists() else {}

    if "artifacts" in updates:
        merged = {item["path"]: item for item in data.get("artifacts", [])}
        merged.update({item["path"]: item for item in updates["artifacts"]})
        updates["artifacts"] = [merged[name] for name in sorted(merged)]

    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(
    case: Case,
    run_directory,
    *,
    solvers=("openmc", "opensn"),
) -> Path:
    """Generate selected solver inputs without starting a subprocess."""
    if not isinstance(case, Case):
        raise TypeError("case must be a Case")

    if isinstance(solvers, (str, bytes)):
        raise ValueError("solvers must be an iterable of solver names")
    try:
        requested = tuple(solvers)
    except TypeError as error:
        raise ValueError("solvers must be an iterable of solver names") from error

    if not requested:
        raise ValueError("at least one solver must be selected")
    if any(not isinstance(name, str) for name in requested):
        raise ValueError("solver names must be strings")
    if len(set(requested)) != len(requested):
        raise ValueError("solver names must not be duplicated")

    unknown = set(requested) - {"openmc", "opensn"}
    if unknown:
        raise ValueError(f"unknown solver {sorted(unknown)[0]!r}")
    if "opensn" in requested and "openmc" not in requested:
        raise ValueError("OpenSn preparation requires OpenMC")

    # Store and generate in dependency order even when the caller supplies an
    # unordered iterable such as a set.
    selected = tuple(name for name in ("openmc", "opensn") if name in requested)

    from .openmc import _write_openmc_input

    # Each prepared directory is self-contained; generation never depends on
    # or updates shared campaign state.
    run = Path(run_directory).resolve()
    openmc_input = run / "openmc" / "model.py"
    openmc_input.parent.mkdir(parents=True, exist_ok=True)
    _write_openmc_input(case, openmc_input)

    artifacts = [_artifact(openmc_input, run)]
    if "opensn" in selected:
        from .opensn import _write_opensn_input

        opensn_input = run / "opensn" / "input.py"
        opensn_input.parent.mkdir(parents=True, exist_ok=True)
        _write_opensn_input(case, opensn_input)
        artifacts.append(_artifact(opensn_input, run))

    from . import __version__

    _update_run_metadata(
        run,
        case=_jsonable_case(case),
        generate_mgxs_version=__version__,
        solvers=list(selected),
        artifacts=artifacts,
    )

    return run
