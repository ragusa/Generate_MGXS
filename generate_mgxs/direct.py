"""The homogeneous fixed-source balance solve."""

from __future__ import annotations

import numpy as np

from .mgxs import MGXS
from .results import InfiniteMediumSolution, Spectrum


def solve_infinite_medium(
    mgxs: MGXS, source_probabilities, volume_cm3: float
) -> InfiniteMediumSolution:
    """Solve a non-fissionable homogeneous fixed-source balance equation."""
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
    if source.shape != (groups,) or np.any(source < 0.0) or not np.isclose(source.sum(), 1.0):
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
