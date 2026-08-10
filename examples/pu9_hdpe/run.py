"""Prepare and run the managed legacy Pu9+HDPE eigenvalue case with OpenMC."""

import os
from pathlib import Path

from generate_mgxs import prepare, run_openmc

from case import CASE


run_path = prepare(CASE, Path("run/pu9_hdpe"), solvers=("openmc",))
run_openmc(
    run_path,
    cross_sections=Path(os.environ["OPENMC_CROSS_SECTIONS"]),
)
