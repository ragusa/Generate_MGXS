"""Small result value objects exposed by the package."""

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


class EigenvalueSolution:
    """Normalized direct eigenvector, multiplication factor, and residual."""

    def __init__(self, spectrum, k_eff, residual):
        self.spectrum = spectrum
        self.k_eff = float(k_eff)
        self.residual = float(residual)


class OpenMCEigenvalueResult:
    """OpenMC eigenvalue estimate and its raw group-integrated flux tally."""

    def __init__(self, spectrum, k_eff, k_eff_std_dev):
        self.spectrum = spectrum
        self.k_eff = float(k_eff)
        self.k_eff_std_dev = float(k_eff_std_dev)

        if not np.isfinite(self.k_eff) or self.k_eff <= 0.0:
            raise ValueError("OpenMC k_eff must be finite and positive")
        if not np.isfinite(self.k_eff_std_dev) or self.k_eff_std_dev < 0.0:
            raise ValueError("OpenMC k_eff_std_dev must be finite and nonnegative")


class OpenSnResult:
    """A strictly converged fixed-source or eigenvalue OpenSn solution."""

    def __init__(
        self,
        spectrum,
        converged,
        iterations=None,
        residual=None,
        balance=None,
        domain_spectra=None,
        *,
        run_mode="fixed_source",
        k_eff=None,
        k_eff_change=None,
        power_iterations=None,
        sweeps=None,
    ):
        self.spectrum = spectrum
        self.converged = bool(converged)
        self.iterations = None if iterations is None else int(iterations)
        self.residual = None if residual is None else float(residual)
        self.balance = None if balance is None else float(balance)
        self.domain_spectra = domain_spectra
        self.run_mode = run_mode
        self.k_eff = None if k_eff is None else float(k_eff)
        self.k_eff_change = (
            None if k_eff_change is None else float(k_eff_change)
        )
        self.power_iterations = (
            None if power_iterations is None else int(power_iterations)
        )
        self.sweeps = None if sweeps is None else int(sweeps)

        if self.run_mode not in {"fixed_source", "eigenvalue"}:
            raise ValueError("OpenSn run_mode must be fixed_source or eigenvalue")
        if self.run_mode == "eigenvalue":
            if (
                self.k_eff is None
                or not np.isfinite(self.k_eff)
                or self.k_eff <= 0.0
            ):
                raise ValueError("OpenSn k_eff must be finite and positive")
            if (
                self.k_eff_change is None
                or not np.isfinite(self.k_eff_change)
                or self.k_eff_change < 0.0
            ):
                raise ValueError("OpenSn k_eff_change must be finite and nonnegative")
            if self.power_iterations is None or self.power_iterations < 1:
                raise ValueError("OpenSn power_iterations must be positive")
            if self.sweeps is None or self.sweeps < 1:
                raise ValueError("OpenSn sweeps must be positive")
