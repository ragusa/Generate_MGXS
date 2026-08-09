"""The three result types exposed by the package."""

from __future__ import annotations

import numpy as np


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    return array


class Spectrum:
    """A group-integrated spectrum in ascending physical-energy order."""

    def __init__(self, energy_bounds_ev, values, std_dev=None, logical_domain=None):
        bounds = _vector(energy_bounds_ev, "energy_bounds_ev")
        values = _vector(values, "values")
        if bounds.size != values.size + 1 or np.any(np.diff(bounds) <= 0.0):
            raise ValueError("spectrum boundaries must be ascending with G + 1 values")
        std = None
        if std_dev is not None:
            std = _vector(std_dev, "std_dev")
            if std.shape != values.shape or np.any(std < 0.0):
                raise ValueError("standard deviations must be nonnegative and match values")
        self.energy_bounds_ev = bounds
        self.values = values
        self.std_dev = std
        self.logical_domain = logical_domain

    @property
    def normalized(self) -> np.ndarray:
        """Return values divided by their group sum."""
        total = float(np.sum(self.values))
        if total == 0.0:
            raise ValueError("cannot normalize a zero spectrum")
        return np.asarray(self.values) / total


class InfiniteMediumSolution:
    """Direct-solver spectrum, flux density, residual, and global balance."""

    def __init__(self, spectrum, flux_density, residual, balance):
        self.spectrum = spectrum
        self.flux_density = np.asarray(flux_density, dtype=float)
        self.residual = float(residual)
        self.balance = float(balance)


class OpenSnResult:
    """A strictly converged OpenSn solution and optional material spectra."""

    def __init__(
        self, spectrum, converged, iterations, residual, balance=None, domain_spectra=None
    ):
        self.spectrum = spectrum
        self.converged = bool(converged)
        self.iterations = int(iterations)
        self.residual = float(residual)
        self.balance = None if balance is None else float(balance)
        self.domain_spectra = domain_spectra
