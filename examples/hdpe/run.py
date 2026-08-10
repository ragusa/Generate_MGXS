"""Prepare, execute, and compare HDPE using environment-configured solvers."""

import os
from pathlib import Path

from generate_mgxs import (
    load_mgxs,
    load_openmc_result,
    plot_mgxs,
    plot_spectra,
    prepare,
    run_openmc,
    run_opensn,
    solve_infinite_medium,
)
from case import CASE, HDPE


# --- Preparation: generates files but starts no solver --------------------
run_path = prepare(CASE, Path("run"))
print(f"Prepared {run_path}; generated inputs may instead be submitted externally.")

# --- OpenMC transport -----------------------------------------------------
run_openmc(
    run_path,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="run",
)

# --- OpenMC statepoint and MGXS processing -------------------------------
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

# --- MGXS plots -----------------------------------------------------------
plot_mgxs(mgxs, output_directory=run_path / "plots")

# --- Homogeneous direct solution -----------------------------------------
direct = solve_infinite_medium(
    mgxs,
    CASE.source_probabilities,
    CASE.source_volume_cm3,
)

# --- OpenSn verification --------------------------------------------------
opensn = run_opensn(run_path, executable=os.environ["OPENSN_CONSOLE"])

# --- Result comparison ----------------------------------------------------
plot_spectra(
    openmc_result,
    opensn.spectrum,
    direct.spectrum,
    output_directory=run_path / "plots",
)
