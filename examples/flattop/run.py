"""Prepare, execute, and compare the homogeneous FlatTop eigenvalue case."""

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
    solve_infinite_medium_eigenvalue,
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


run_path = prepare(CASE, Path("run"))
STAGES["Preparation"] = "PASSED"

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
    openmc = _attempt("OpenMC result load", lambda: load_openmc_result(run_path))
    mgxs = _attempt(
        "MGXS load",
        lambda: load_mgxs(run_path / "openmc/mgxs.h5", "flattop_nu", 294.0),
    )
else:
    openmc = None
    mgxs = None
    _skip("OpenMC result load", "OpenMC processing is unavailable")
    _skip("MGXS load", "OpenMC processing is unavailable")

if mgxs is not None:
    direct = _attempt(
        "Direct eigenvalue solve",
        lambda: solve_infinite_medium_eigenvalue(mgxs),
    )
    _attempt(
        "MGXS plots",
        lambda: plot_mgxs(mgxs, output_directory=run_path / "plots"),
    )
else:
    direct = None
    _skip("Direct eigenvalue solve", "MGXS data is unavailable")
    _skip("MGXS plots", "MGXS data is unavailable")

if processing is not None and (run_path / "openmc/mgxs.h5").is_file():
    opensn = _attempt(
        "OpenSn",
        lambda: run_opensn(run_path, executable=os.environ["OPENSN_CONSOLE"]),
    )
else:
    opensn = None
    _skip("OpenSn", "the processed MGXS file is unavailable")

if openmc is None:
    print("OpenMC k_eff = unavailable")
else:
    print(f"OpenMC k_eff = {openmc.k_eff:.8f} +/- {openmc.k_eff_std_dev:.2e}")
print(
    "Direct  k_eff = unavailable"
    if direct is None
    else f"Direct  k_eff = {direct.k_eff:.8f}"
)
print(
    "OpenSn  k_eff = unavailable"
    if opensn is None
    else f"OpenSn  k_eff = {opensn.k_eff:.8f}"
)
if direct is not None and openmc is not None:
    print(f"Direct - OpenMC = {direct.k_eff - openmc.k_eff:+.8e}")
if opensn is not None and direct is not None:
    print(f"OpenSn - Direct = {opensn.k_eff - direct.k_eff:+.8e}")

spectra = {
    "openmc": None if openmc is None else openmc.spectrum,
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
