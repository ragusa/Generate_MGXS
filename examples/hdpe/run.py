"""Prepare, execute, and compare HDPE using environment-configured solvers."""

import os
from pathlib import Path

from generate_mgxs import (
    load_mgxs, load_openmc_result, plot_mgxs, plot_spectra, prepare,
    run_openmc, run_opensn, solve_infinite_medium,
)
from case import CASE, HDPE


# --- Preparation: generates files but starts no solver --------------------
# prepare() writes independent inputs and metadata only; it starts no process.
run_path = prepare(CASE, Path("run/hdpe"))
print(f"Prepared {run_path}; generated inputs may instead be submitted externally.")

# --- OpenMC transport -----------------------------------------------------
run_openmc(
    run_path,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="run",
)


# --- OpenMC statepoint/MGXS processing -----------------------------------
run_openmc(
    run_path,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="process",
)
openmc_result = load_openmc_result(run_path)
mgxs = load_mgxs(
    run_path / "openmc/mgxs.h5",
    HDPE.logical_name,
    HDPE.temperature_k,
)

# MGXS plotting is explicit postprocessing; it neither reads a statepoint nor
# invokes a solver.
plot_mgxs(mgxs, output_directory=run_path / "plots")


# --- Optional homogeneous direct solve -----------------------------------
direct = solve_infinite_medium(mgxs, CASE.source_probabilities, CASE.source_volume_cm3)


# --- OpenSn verification --------------------------------------------------
# OpenSn independently places HDPE in a fixed reflected 8-cell cube. It does
# not reproduce the long OpenMC tally geometry and uses P0 for this scalar-flux
# comparison.
opensn = run_opensn(run_path, executable=os.environ["OPENSN_CONSOLE"])


# --- Result comparison ----------------------------------------------------
# Expected solver products include openmc/mgxs.h5, both compact result JSON
# files, diagnostic uncertainty JSON, persistent logs, and comparison plots.
plot_spectra(
    openmc_result,
    opensn.spectrum,
    direct.spectrum,
    output_directory=run_path / "plots",
)
