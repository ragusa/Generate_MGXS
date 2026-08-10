"""Public API for transparent OpenMC/OpenSn multigroup calculations."""

from .case import Case, Material, energy_bounds, prepare, source_probabilities
from .direct import solve_infinite_medium, solve_infinite_medium_eigenvalue
from .mgxs import MGXS, load_mgxs
from .openmc import load_openmc_result, run_openmc
from .opensn import load_opensn_result, run_opensn
from .plotting import plot_mgxs, plot_spectra
from .results import (
    EigenvalueSolution,
    InfiniteMediumSolution,
    OpenMCEigenvalueResult,
    OpenSnResult,
    Spectrum,
)

__version__ = "0.1.0"

__all__ = [
    "Case",
    "EigenvalueSolution",
    "InfiniteMediumSolution",
    "MGXS",
    "Material",
    "OpenMCEigenvalueResult",
    "OpenSnResult",
    "Spectrum",
    "energy_bounds",
    "load_mgxs",
    "load_openmc_result",
    "load_opensn_result",
    "plot_mgxs",
    "plot_spectra",
    "prepare",
    "run_openmc",
    "run_opensn",
    "solve_infinite_medium",
    "solve_infinite_medium_eigenvalue",
    "source_probabilities",
]
