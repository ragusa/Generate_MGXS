"""Prepare and run the UO2/HDPE example with portable environment settings."""

import os
from pathlib import Path

from generate_mgxs import load_mgxs, load_openmc_result, prepare, run_openmc, run_opensn
from case import CASE


# --- Preparation: generates files but starts no solver --------------------
# Generation is safe for bulk loops: this call writes files but executes no solver.
run = prepare(CASE, Path("run/uo2_in_hdpe"))
print(f"Prepared independent run directory {run}")

# --- OpenMC transport -----------------------------------------------------
# Execution remains explicit and may instead be performed later by Bash or SLURM.
run_openmc(
    run,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="run",
)


# --- OpenMC statepoint/MGXS processing -----------------------------------
run_openmc(
    run,
    cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    operation="process",
)
openmc_result = load_openmc_result(run)
target_xs = load_mgxs(run / "openmc/mgxs.h5", "uo2_target", 294.0)
moderator_xs = load_mgxs(run / "openmc/mgxs.h5", "hdpe_moderator", 294.0)


# --- OpenSn execution and result summary ---------------------------------
opensn = run_opensn(run, executable=os.environ["OPENSN_CONSOLE"])

print(openmc_result.values.sum(), target_xs.logical_domain, moderator_xs.logical_domain)
print({name: spectrum.values.sum() for name, spectrum in opensn.domain_spectra.items()})
