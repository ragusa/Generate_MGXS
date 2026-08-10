"""Prepare, execute, and compare Be-9 using environment-configured solvers."""

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
    run_opensn,
    solve_infinite_medium,
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


# Preparation is foundational and intentionally remains fail-fast.
run_path = prepare(CASE, Path("run"))
STAGES["Preparation"] = "PASSED"
print(f"Prepared {run_path}; generated inputs may instead be submitted externally.")

transport = _attempt(
    "OpenMC transport",
    lambda: run_openmc(
        run_path,
        cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
        operation="run",
    ),
)
statepoints = tuple((run_path / "openmc").glob("statepoint.*.h5"))
if transport is not None or statepoints:
    processing = _attempt(
        "OpenMC processing",
        lambda: run_openmc(
            run_path,
            cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
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
    mgxs = _attempt(
        "MGXS load",
        lambda: load_mgxs(run_path / "openmc/mgxs.h5", "be9", 294.0),
    )
else:
    openmc_result = None
    mgxs = None
    _skip("OpenMC result load", "OpenMC processing is unavailable")
    _skip("MGXS load", "OpenMC processing is unavailable")

if mgxs is not None:
    direct = _attempt(
        "Direct fixed-source solve",
        lambda: solve_infinite_medium(
            mgxs, CASE.source_probabilities, CASE.source_volume_cm3
        ),
    )
    _attempt(
        "MGXS plots",
        lambda: plot_mgxs(mgxs, output_directory=run_path / "plots"),
    )
else:
    direct = None
    _skip("Direct fixed-source solve", "MGXS data is unavailable")
    _skip("MGXS plots", "MGXS data is unavailable")

if processing is not None and (run_path / "openmc/mgxs.h5").is_file():
    opensn = _attempt(
        "OpenSn",
        lambda: run_opensn(run_path, executable=os.environ["OPENSN_CONSOLE"]),
    )
else:
    opensn = None
    _skip("OpenSn", "the processed MGXS file is unavailable")

spectra = {
    "openmc": openmc_result,
    "opensn": None if opensn is None else opensn.spectrum,
    "direct": None if direct is None else direct.spectrum,
}
included = tuple(name for name, spectrum in spectra.items() if spectrum is not None)
if included:
    _attempt(
        "Spectrum plots",
        lambda: plot_spectra(
            spectra["openmc"],
            spectra["opensn"],
            spectra["direct"],
            include=included,
            output_directory=run_path / "plots",
        ),
    )
else:
    _skip("Spectrum plots", "no successful spectrum is available")

_print_summary()
