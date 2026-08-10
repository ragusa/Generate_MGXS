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


run_path = prepare(CASE, Path("run"), solvers=("openmc",))
run_openmc(
    run_path,
    cross_sections=Path(os.environ["OPENMC_CROSS_SECTIONS"]),
    operation="all",
)

openmc_result = load_openmc_result(run_path)
mgxs_path = run_path / "openmc/mgxs.h5"
plots = run_path / "plots"

# Each cell owns a distinct XSdata name even when cells share one physical
# material. plot_mgxs prefixes filenames with that logical domain name.
for cell in CASE.geometry.domains:
    mgxs = load_mgxs(
        mgxs_path,
        cell.xsdata_name,
        cell.material.temperature_k,
    )
    plot_mgxs(mgxs, output_directory=plots)

# This geometry has no matching direct or OpenSn calculation. The plotting API
# supports selecting the OpenMC spectrum alone without inventing a comparison.
plot_spectra(
    openmc_result,
    None,
    None,
    include=("openmc",),
    output_directory=plots,
)

# Compare independently normalized spectral shapes in every cell domain. The
# XSdata name, not physical material identity, keeps Hdpe and Outer distinct.
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
