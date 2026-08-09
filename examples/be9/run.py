"""Prepare, execute, and compare Be-9 using environment-configured solvers."""

import os
from pathlib import Path

from generate_mgxs import (
    load_mgxs, load_openmc_result, plot_spectra, prepare, run_openmc,
    run_opensn, solve_infinite_medium,
)
from case import CASE


# prepare() writes independent inputs and metadata only; it starts no process.
run = prepare(CASE, Path("run/be9"))
print(f"Prepared {run}; generated inputs may instead be submitted externally.")

# These calls are the actions that actually execute OpenMC and OpenSn.
run_openmc(run, cross_sections=os.environ["OPENMC_CROSS_SECTIONS"])
openmc_result = load_openmc_result(run)
mgxs = load_mgxs(run / "openmc/mgxs.h5", "be9", 294.0)
direct = solve_infinite_medium(mgxs, CASE.source_probabilities, CASE.source_volume_cm3)
opensn = run_opensn(run, executable=os.environ["OPENSN_CONSOLE"])
# Expected solver products include openmc/mgxs.h5, both compact result JSON
# files, diagnostic uncertainty JSON, persistent logs, and this comparison plot.
(run / "plots").mkdir(exist_ok=True)
plot_spectra(
    openmc_result, direct.spectrum, opensn.spectrum,
    labels=("OpenMC", "direct", "OpenSn"), path=run / "plots/spectra.png",
)
