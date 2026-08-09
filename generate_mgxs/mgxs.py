"""Load OpenMC MGXS HDF5 into one canonical scientific representation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


class MGXS:
    """Macroscopic data with ``scatter[moment, g_in, g_out]`` ordering."""

    def __init__(
        self,
        energy_bounds_ev,
        total,
        absorption,
        scatter,
        logical_domain: str,
        temperature_k: float,
        fission=None,
        nu_fission=None,
        chi_raw=None,
        chi=None,
        chi_raw_sum=None,
        chi_normalization_factor=None,
        chi_closure_correction=None,
        uncertainty=None,
    ):
        self.energy_bounds_ev = np.asarray(energy_bounds_ev, dtype=float)
        self.total = np.asarray(total, dtype=float)
        self.absorption = np.asarray(absorption, dtype=float)
        self.scatter = np.asarray(scatter, dtype=float)
        self.logical_domain = logical_domain
        self.temperature_k = float(temperature_k)
        self.fission = None if fission is None else np.asarray(fission, dtype=float)
        self.nu_fission = (
            None if nu_fission is None else np.asarray(nu_fission, dtype=float)
        )
        self.chi_raw = None if chi_raw is None else np.asarray(chi_raw, dtype=float)
        self.chi = None if chi is None else np.asarray(chi, dtype=float)
        self.chi_raw_sum = chi_raw_sum
        self.chi_normalization_factor = chi_normalization_factor
        self.chi_closure_correction = chi_closure_correction
        self.uncertainty = uncertainty


def _finite_vector(dataset, groups: int, name: str) -> np.ndarray:
    values = np.asarray(dataset, dtype=float)

    if values.shape != (groups,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite {groups}-group vector")

    return values


def _normalize_chi(raw: np.ndarray) -> tuple[np.ndarray, float, float, tuple[int, float]]:
    """Normalize raw OpenMC chi and record the floating-point closure step."""
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0 or np.any(raw < 0.0):
        raise ValueError("raw chi must be finite, nonnegative, and have a positive sum")

    factor = 1.0 / total
    normalized = raw * factor

    # Apply roundoff closure to the largest component so a tiny correction has
    # the smallest relative effect on the physical spectrum.
    index = int(np.argmax(normalized))
    correction = float(1.0 - np.sum(normalized))
    normalized[index] += correction

    return normalized, total, factor, (index, correction)


def _dense_scatter(group, groups: int, moments: int) -> np.ndarray:
    """Expand OpenMC's compressed high-to-low scatter matrix to canonical order."""
    # OpenMC stores only each incident group's populated outgoing range. Read
    # those bounds before reconstructing the dense scientific tensor.
    low = np.asarray(group["g_min"], dtype=int)
    high = np.asarray(group["g_max"], dtype=int)
    flat = np.asarray(group["scatter_matrix"], dtype=float)

    if low.shape != (groups,) or high.shape != (groups,):
        raise ValueError("compressed scatter bounds must contain one range per incident group")

    dense_hilo = np.zeros((moments, groups, groups))
    offset = 0

    for incident, (first, last) in enumerate(zip(low, high)):
        if first < 1 or last < first or last > groups:
            raise ValueError("compressed scatter contains an invalid outgoing range")
        count = int(last - first + 1)
        stop = offset + count * moments
        if stop > flat.size:
            raise ValueError("compressed scatter matrix is truncated")
        dense_hilo[:, incident, first - 1 : last] = flat[offset:stop].reshape(
            count, moments
        ).T
        offset = stop

    if offset != flat.size or not np.all(np.isfinite(dense_hilo)):
        raise ValueError("compressed scatter matrix has trailing or non-finite data")

    # OpenMC stores both incident and outgoing axes high-to-low.  The package
    # contract is low-to-high while retaining [moment, g_in, g_out] orientation.
    return dense_hilo[:, ::-1, ::-1]


def _load_uncertainty(path: Path | None, domain: str, groups: int, moments: int):
    if path is None or not path.is_file():
        return None

    document = json.loads(path.read_text())
    quantities = document.get("domains", {}).get(domain, {}).get("quantities", {})
    result = {}

    for name, record in quantities.items():
        values = np.asarray(record["std_dev"], dtype=float)
        expected = (moments, groups, groups) if name == "scatter" else (groups,)
        if values.shape != expected or np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {name} uncertainty for domain {domain!r}")
        result[name] = values

    return result


def load_mgxs(path, logical_domain: str, temperature_k: float = 294.0, *, uncertainty_path=None) -> MGXS:
    """Load one OpenMC MGXS material domain into the canonical ascending order."""
    path = Path(path)

    with h5py.File(path, "r") as h5:
        # Validate the library-level contract before selecting material data.
        filetype = h5.attrs.get("filetype")
        if isinstance(filetype, bytes):
            filetype = filetype.decode()
        if filetype != "mgxs":
            raise ValueError(f"{path} is not an OpenMC MGXS library")

        groups = int(h5.attrs["energy_groups"])
        bounds = _finite_vector(h5.attrs["group structure"], groups + 1, "group structure")
        if np.any(np.diff(bounds) <= 0.0):
            raise ValueError("MGXS group structure must be strictly ascending in eV")

        # Logical names, rather than incidental OpenMC integer IDs, identify
        # the material domain throughout the handoff.
        if logical_domain not in h5:
            raise KeyError(f"logical domain {logical_domain!r} is absent from {path}")
        domain = h5[logical_domain]

        label = f"{int(round(temperature_k))}K"
        if label not in domain or label not in domain.get("kTs", {}):
            raise KeyError(f"temperature {label} is absent from domain {logical_domain!r}")

        expected_kt = temperature_k * 8.617333262145e-5
        if not np.isclose(float(domain["kTs"][label][()]), expected_kt, rtol=2e-6):
            raise ValueError(f"temperature metadata for {logical_domain!r} is inconsistent")

        data = domain[label]

        # Scalar vectors in OpenMC MGXS HDF5 are high-to-low; reverse them at
        # this handoff so all subsequent numerical code sees ascending energy.
        total = _finite_vector(data["total"], groups, "total")[::-1]
        absorption = _finite_vector(data["absorption"], groups, "absorption")[::-1]

        # Scattering needs both dense reconstruction and reversal of its
        # incident/outgoing axes; _dense_scatter performs those together.
        moments = int(domain.attrs["order"]) + 1
        scatter = _dense_scatter(data["scatter_data"], groups, moments)

        # Fissionability metadata and datasets must agree so downstream code
        # cannot silently omit or invent production physics.
        fissionable = bool(domain.attrs.get("fissionable", False))
        fission_names = {"fission", "nu-fission", "chi"}
        present = fission_names.intersection(data.keys())
        if fissionable and present != fission_names:
            missing = ", ".join(sorted(fission_names - present))
            raise ValueError(f"fissionable domain {logical_domain!r} is missing {missing}")
        if not fissionable and present:
            raise ValueError(
                f"non-fissionable domain {logical_domain!r} contains fission data"
            )

        optional = {}
        for hdf5_name, field in (("fission", "fission"), ("nu-fission", "nu_fission")):
            optional[field] = (
                _finite_vector(data[hdf5_name], groups, hdf5_name)[::-1]
                if hdf5_name in data
                else None
            )
        chi_raw = _finite_vector(data["chi"], groups, "chi")[::-1] if "chi" in data else None

    chi = chi_sum = chi_factor = chi_correction = None
    if chi_raw is not None:
        # Preserve raw chi and its uncertainty.  Normalization changes both the
        # values and covariance, so raw uncertainty is never relabeled as chi.
        chi, chi_sum, chi_factor, chi_correction = _normalize_chi(chi_raw.copy())

    # By default the uncertainty sidecar sits beside the OpenMC products in
    # diagnostics/, but callers may supply an explicit scientific dataset.
    if uncertainty_path is None:
        candidate = path.parent.parent / "diagnostics" / "mgxs_uncertainty.json"
        uncertainty_path = candidate if candidate.is_file() else None
    uncertainty = _load_uncertainty(
        Path(uncertainty_path) if uncertainty_path else None,
        logical_domain,
        groups,
        moments,
    )

    return MGXS(
        bounds,
        total,
        absorption,
        scatter,
        logical_domain,
        float(temperature_k),
        chi_raw=chi_raw,
        chi=chi,
        chi_raw_sum=chi_sum,
        chi_normalization_factor=chi_factor,
        chi_closure_correction=chi_correction,
        uncertainty=uncertainty,
        **optional,
    )
