"""Direct homogeneous fixed-source and rank-one eigenvalue solves."""

from __future__ import annotations

import numpy as np

from .mgxs import MGXS
from .results import EigenvalueSolution, InfiniteMediumSolution, Spectrum


def solve_infinite_medium(
    mgxs: MGXS, source_probabilities, volume_cm3: float
) -> InfiniteMediumSolution:
    """Solve a non-fissionable homogeneous fixed-source balance equation."""
    # This equation has no fission-production term. Reject fissionable data
    # explicitly instead of returning a plausible but incomplete solution.
    if any(
        value is not None
        for value in (mgxs.fission, mgxs.nu_fission, mgxs.chi_raw, mgxs.chi)
    ):
        raise ValueError(
            "the direct solver supports non-fissionable homogeneous MGXS only"
        )

    source = np.asarray(source_probabilities, dtype=float)
    groups = mgxs.total.size

    if volume_cm3 <= 0.0:
        raise ValueError("volume_cm3 must be positive")
    if (
        source.shape != (groups,)
        or np.any(source < 0.0)
        or not np.isclose(source.sum(), 1.0)
    ):
        raise ValueError("source_probabilities must be a normalized G-vector")
    if mgxs.scatter.ndim != 3 or mgxs.scatter.shape[1:] != (groups, groups):
        raise ValueError("scatter must have shape [moment, g_in, g_out]")

    # scatter is [moment, g_in, g_out].  Transposing P0 makes each matrix row
    # collect production into its outgoing group in
    # [diag(Sigma_t) - scatter[0].T] phi = q / volume.
    operator = np.diag(mgxs.total) - mgxs.scatter[0].T
    rhs = source / volume_cm3
    density = np.linalg.solve(operator, rhs)

    # The equation uses a spatial source density; reported spectra are volume
    # integrated so homogeneous results do not depend on the chosen box volume.
    integrated = density * volume_cm3
    residual = float(np.linalg.norm(operator @ density - rhs))
    balance = float(np.sum(operator @ integrated) - np.sum(source))

    return InfiniteMediumSolution(
        Spectrum(mgxs.energy_bounds_ev, integrated, logical_domain=mgxs.logical_domain),
        density,
        residual,
        balance,
    )


def solve_infinite_medium_eigenvalue(mgxs: MGXS) -> EigenvalueSolution:
    """Solve a homogeneous factorized-fission k-eigenvalue problem."""
    groups = mgxs.total.size
    required = {
        "fission": mgxs.fission,
        "nu_fission": mgxs.nu_fission,
        "chi": mgxs.chi,
    }
    missing = [name for name, values in required.items() if values is None]
    if missing:
        raise ValueError(
            "eigenvalue solve requires fission, nu_fission, and normalized chi"
        )

    vectors = {
        "total": np.asarray(mgxs.total, dtype=float),
        **{name: np.asarray(values, dtype=float) for name, values in required.items()},
    }
    for name, values in vectors.items():
        if values.shape != (groups,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite G-vector")
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be nonnegative")
    if not np.isclose(vectors["chi"].sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("chi must be normalized before the eigenvalue solve")
    if mgxs.scatter.ndim != 3 or mgxs.scatter.shape[1:] != (groups, groups):
        raise ValueError("scatter must have shape [moment, g_in, g_out]")
    if not np.all(np.isfinite(mgxs.scatter)):
        raise ValueError("scatter must be finite")

    # Canonical scatter rows are incoming groups. Transposition gives the
    # ordinary balance operator whose rows collect production into g_out.
    operator = np.diag(vectors["total"]) - mgxs.scatter[0].T
    try:
        y = np.linalg.solve(operator, vectors["chi"])
    except np.linalg.LinAlgError as error:
        raise ValueError("eigenvalue loss operator is singular") from error

    k_eff = float(vectors["nu_fission"] @ y)
    if not np.isfinite(k_eff) or k_eff <= 0.0:
        raise ValueError("eigenvalue solve produced a nonpositive or non-finite k_eff")
    if not np.all(np.isfinite(y)):
        raise ValueError("eigenvalue solve produced a non-finite flux")

    # Roundoff can place a mathematically zero component just below zero.
    # Reject physically negative vectors, but close only machine-scale noise.
    negative_tolerance = 1.0e-13 * max(1.0, float(np.max(np.abs(y))))
    if np.any(y < -negative_tolerance):
        raise ValueError("eigenvalue solve produced a negative flux")
    y = np.where(y < 0.0, 0.0, y)
    normalization = float(y.sum())
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("eigenvalue solve produced a zero flux")
    phi = y / normalization

    # F = outer(chi, nu_fission) in [g_out, g_in] linear-algebra ordering.
    residual_vector = operator @ phi - (
        vectors["chi"] * (vectors["nu_fission"] @ phi) / k_eff
    )
    residual = float(np.linalg.norm(residual_vector))
    if not np.isfinite(residual):
        raise ValueError("eigenvalue solve produced a non-finite residual")

    return EigenvalueSolution(
        Spectrum(
            mgxs.energy_bounds_ev,
            phi,
            logical_domain=mgxs.logical_domain,
        ),
        k_eff,
        residual,
    )
