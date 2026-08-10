"""Prepare, run, and plot the managed legacy detector with OpenMC only."""

import os
from pathlib import Path

from generate_mgxs import (
    load_mgxs,
    load_openmc_domain_spectra,
    load_openmc_result,
    plot_mgxs,
    plot_openmc_domain_spectra,
    plot_spectra,
    prepare,
    run_openmc,
)
from case import CASE


# --- Preparation ----------------------------------------------------------
run_path = prepare(CASE, Path("run"), solvers=("openmc",))

# --- OpenMC transport and MGXS processing --------------------------------
run_openmc(
    run_path,
    cross_sections=Path(os.environ["OPENMC_CROSS_SECTIONS"]),
    operation="all",
)
openmc_result = load_openmc_result(run_path)
mgxs_path = run_path / "openmc/mgxs.h5"
plots = run_path / "plots"

# --- Per-domain MGXS plots ------------------------------------------------
for cell in CASE.geometry.domains:
    mgxs = load_mgxs(
        mgxs_path,
        cell.xsdata_name,
        cell.material.temperature_k,
    )
    plot_mgxs(mgxs, output_directory=plots)

# --- Primary OpenMC spectrum ---------------------------------------------
plot_spectra(
    openmc_result,
    None,
    None,
    include=("openmc",),
    output_directory=plots,
)

# --- All-domain OpenMC spectra -------------------------------------------
domain_spectra = load_openmc_domain_spectra(run_path)
domain_labels = {
    cell.xsdata_name: cell.name
    for cell in CASE.geometry.domains
}
plot_openmc_domain_spectra(
    domain_spectra,
    labels=domain_labels,
    output_directory=plots,
)
