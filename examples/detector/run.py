"""Prepare, run, and plot the managed legacy detector with OpenMC only."""

import os
from pathlib import Path
import warnings

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

if openmc_result is not None:
    _attempt(
        "Spectrum plots",
        lambda: plot_spectra(
            openmc_result,
            None,
            None,
            include=("openmc",),
            output_directory=plots,
        ),
    )
else:
    _skip("Spectrum plots", "no successful spectrum is available")

if processing is not None:
    domain_spectra = _attempt(
        "Domain spectra load", lambda: load_openmc_domain_spectra(run_path)
    )
else:
    domain_spectra = None
    _skip("Domain spectra load", "OpenMC processing is unavailable")

if domain_spectra is not None:
    domain_labels = {
        cell.xsdata_name: cell.name
        for cell in CASE.geometry.domains
    }
    _attempt(
        "Domain spectrum plots",
        lambda: plot_openmc_domain_spectra(
            domain_spectra,
            labels=domain_labels,
            output_directory=plots,
        ),
    )
else:
    _skip("Domain spectrum plots", "domain spectra are unavailable")

_print_summary()
