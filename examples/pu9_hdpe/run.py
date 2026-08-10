"""Prepare, run, and plot the managed Pu9+HDPE eigenvalue case with OpenMC."""

import os
from pathlib import Path
import warnings

from generate_mgxs import (
    load_mgxs,
    load_openmc_result,
    plot_mgxs,
    plot_spectra,
    prepare,
    run_openmc,
)
from case import CASE


STAGES = {}


def _attempt(stage, action):
    try:
        result = action()
    except Exception as error:
        warnings.warn(
            f"{stage} failed: {error}\nContinuing with remaining stages.",
            RuntimeWarning,
            stacklevel=2,
        )
        STAGES[stage] = "FAILED"
        return None
    STAGES[stage] = "PASSED"
    return result


def _skip(stage, reason):
    warnings.warn(f"{stage} skipped: {reason}", RuntimeWarning, stacklevel=2)
    STAGES[stage] = "SKIPPED"


def _print_summary():
    print("Run summary:")
    for stage, status in STAGES.items():
        print(f"  {stage:<28}: {status}")


run_path = prepare(CASE, Path("run"), solvers=("openmc",))
STAGES["Preparation"] = "PASSED"
plots = run_path / "plots"

transport = _attempt(
    "OpenMC transport",
    lambda: run_openmc(
        run_path,
        cross_sections=Path(os.environ["OPENMC_CROSS_SECTIONS"]),
        operation="run",
    ),
)
statepoints = tuple((run_path / "openmc").glob("statepoint.*.h5"))
if transport is not None or statepoints:
    processing = _attempt(
        "OpenMC processing",
        lambda: run_openmc(
            run_path,
            cross_sections=Path(os.environ["OPENMC_CROSS_SECTIONS"]),
            operation="process",
        ),
    )
else:
    processing = None
    _skip("OpenMC processing", "transport failed and no statepoint is available")

if processing is not None:
    openmc_result = _attempt(
        "OpenMC result load", lambda: load_openmc_result(run_path)
    )
else:
    openmc_result = None
    _skip("OpenMC result load", "OpenMC processing is unavailable")

mgxs_path = run_path / "openmc/mgxs.h5"
for cell in CASE.geometry.domains:
    stage_suffix = cell.xsdata_name
    if processing is None:
        _skip(f"MGXS load ({stage_suffix})", "OpenMC processing is unavailable")
        _skip(f"MGXS plot ({stage_suffix})", "MGXS data is unavailable")
        continue
    mgxs = _attempt(
        f"MGXS load ({stage_suffix})",
        lambda cell=cell: load_mgxs(
            mgxs_path, cell.xsdata_name, cell.material.temperature_k
        ),
    )
    if mgxs is None:
        _skip(f"MGXS plot ({stage_suffix})", "MGXS data is unavailable")
        continue
    _attempt(
        f"MGXS plot ({stage_suffix})",
        lambda mgxs=mgxs: plot_mgxs(mgxs, output_directory=plots),
    )

if openmc_result is None:
    print("OpenMC k_eff = unavailable")
    _skip("Spectrum plots", "no successful spectrum is available")
else:
    print(
        f"OpenMC k_eff = {openmc_result.k_eff:.8f} "
        f"+/- {openmc_result.k_eff_std_dev:.2e}"
    )
    _attempt(
        "Spectrum plots",
        lambda: plot_spectra(
            openmc_result.spectrum,
            None,
            None,
            include=("openmc",),
            output_directory=plots,
        ),
    )

_print_summary()
