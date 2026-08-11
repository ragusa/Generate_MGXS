"""Prepare, run, and plot the UO2/HDPE example with OpenMC only."""

import os
from pathlib import Path

from generate_mgxs import (
    load_mgxs,
    load_openmc_result,
    plot_mgxs,
    prepare,
    run_openmc,
)
from case import CASE


# --- Preparation: generates files but starts no solver --------------------
# Generation is safe for bulk loops: this call writes files but executes no solver.
run_path = prepare(CASE, Path("run/uo2_in_hdpe"), solvers=("openmc",))
print(f"Prepared independent run directory {run_path}")

# --- OpenMC transport -----------------------------------------------------
# Execution remains explicit and may instead be performed later by Bash or SLURM.
run_openmc(
    run_path,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="run",
)


# --- OpenMC statepoint processing and result loading ----------------------
run_openmc(
    run_path,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="process",
)
openmc_result = load_openmc_result(run_path)
target_xs = load_mgxs(run_path / "openmc/mgxs.h5", "uo2_target", 294.0)
moderator_xs = load_mgxs(run_path / "openmc/mgxs.h5", "hdpe_moderator", 294.0)

# --- Per-domain MGXS plotting --------------------------------------------
# The UO2 domain additionally produces chi and the derived fission-production
# matrix. Domain-based filenames let both materials share one output directory.
plot_mgxs(target_xs, output_directory=run_path / "plots")
plot_mgxs(moderator_xs, output_directory=run_path / "plots")

# --- Result summary -------------------------------------------------------
print(openmc_result.values.sum(), target_xs.logical_domain, moderator_xs.logical_domain)
