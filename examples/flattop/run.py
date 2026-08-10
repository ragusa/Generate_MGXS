"""Prepare, execute, and compare the homogeneous FlatTop eigenvalue case."""

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
    solve_infinite_medium_eigenvalue,
)
from case import CASE


# --- Preparation ----------------------------------------------------------
run_path = prepare(CASE, Path("run"))

# --- OpenMC transport and MGXS processing --------------------------------
run_openmc(
    run_path,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="all",
)
openmc = load_openmc_result(run_path)
mgxs = load_mgxs(run_path / "openmc/mgxs.h5", "flattop_nu", 294.0)

# --- Direct eigenvalue verification (disabled: singular loss matrix) -----
# direct = solve_infinite_medium_eigenvalue(mgxs)

# --- OpenSn eigenvalue verification --------------------------------------
opensn = run_opensn(run_path, executable=os.environ["OPENSN_CONSOLE"])

# --- Eigenvalue comparison ------------------------------------------------
print(f"OpenMC k_eff = {openmc.k_eff:.8f} +/- {openmc.k_eff_std_dev:.2e}")
# print(f"Direct  k_eff = {direct.k_eff:.8f}")
print(f"OpenSn  k_eff = {opensn.k_eff:.8f}")
# print(f"Direct - OpenMC = {direct.k_eff - openmc.k_eff:+.8e}")
# print(f"OpenSn - Direct = {opensn.k_eff - direct.k_eff:+.8e}")

# --- MGXS plots -----------------------------------------------------------
plot_mgxs(mgxs, output_directory=run_path / "plots")

# --- Spectrum comparison -------------------------------------------------
# Eigenvectors have arbitrary amplitudes; plot_spectra independently normalizes
# their group-integrated shapes before energy/lethargy comparisons.
plot_spectra(
    openmc.spectrum,
    opensn.spectrum,
    None,
    include=("openmc", "opensn"),
    output_directory=run_path / "plots",
)
