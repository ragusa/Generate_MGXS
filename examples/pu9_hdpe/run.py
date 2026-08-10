"""Prepare, run, and plot the managed Pu9+HDPE eigenvalue case with OpenMC."""

import os
from pathlib import Path

from generate_mgxs import (
    load_mgxs,
    load_openmc_result,
    plot_mgxs,
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

for cell in CASE.geometry.domains:
    mgxs = load_mgxs(
        mgxs_path,
        cell.xsdata_name,
        cell.material.temperature_k,
    )
    plot_mgxs(mgxs, output_directory=plots)

print(
    f"OpenMC k_eff = {openmc_result.k_eff:.8f} "
    f"+/- {openmc_result.k_eff_std_dev:.2e}"
)

# Eigenvector amplitude is arbitrary; plot_spectra normalizes the OpenMC group
# spectrum without introducing a direct or OpenSn comparison.
plot_spectra(
    openmc_result.spectrum,
    None,
    None,
    include=("openmc",),
    output_directory=plots,
)
