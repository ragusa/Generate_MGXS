"""Headless-friendly plots for spectra and multigroup cross sections."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .mgxs import MGXS


def plot_spectra(*spectra, labels=None, path=None, normalize=True):
    """Plot stepwise spectra against ascending energy boundaries."""
    import matplotlib.pyplot as plt

    labels = labels or [
        spectrum.logical_domain or f"spectrum {index + 1}"
        for index, spectrum in enumerate(spectra)
    ]

    figure, axis = plt.subplots()

    # stairs() consumes one more boundary than values, exactly matching the
    # package's group-integrated spectrum representation.
    for spectrum, label in zip(spectra, labels):
        values = spectrum.normalized if normalize else spectrum.values
        axis.stairs(values, spectrum.energy_bounds_ev, label=label)

    axis.set_xscale("log")
    axis.set_xlabel("Energy (eV)")
    axis.set_ylabel("Normalized group flux" if normalize else "Group-integrated flux")
    axis.legend()
    figure.tight_layout()

    if path is not None:
        figure.savefig(path)

    return figure, axis


def _filename_stem(logical_domain: str) -> str:
    """Return a stable, readable filename component for a logical domain."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", logical_domain.strip()).strip("._-")
    return stem.lower() or "mgxs"


def _plot_group_matrix(
    values_gin_gout,
    energy_bounds_ev,
    *,
    title: str,
    colorbar_label: str,
):
    """Plot a canonical ``[g_in, g_out]`` matrix against energy boundaries."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()

    # pcolormesh treats its first data index as vertical.  Transpose only this
    # plotting view so x remains incoming energy and y remains outgoing energy;
    # the canonical scientific array itself stays [g_in, g_out].
    mesh = axis.pcolormesh(
        energy_bounds_ev,
        energy_bounds_ev,
        np.asarray(values_gin_gout).T,
        shading="flat",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Incoming energy [eV]")
    axis.set_ylabel("Outgoing energy [eV]")
    axis.set_title(title)

    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()

    return figure


def plot_mgxs(
    mgxs: MGXS,
    *,
    output_directory=None,
    scatter_moments=(0,),
    show=False,
):
    """Plot one loaded MGXS domain without reading files or running a solver."""
    if not isinstance(mgxs, MGXS):
        raise TypeError("mgxs must be an MGXS object")

    try:
        moments = tuple(scatter_moments)
    except TypeError as error:
        raise ValueError("scatter_moments must be an iterable of integers") from error

    for moment in moments:
        if (
            isinstance(moment, bool)
            or not isinstance(moment, (int, np.integer))
            or moment < 0
            or moment >= mgxs.scatter.shape[0]
        ):
            raise ValueError(f"invalid scattering moment {moment!r}")
    if len(set(moments)) != len(moments):
        raise ValueError("scattering moments must not be duplicated")

    fission_fields = (mgxs.fission, mgxs.nu_fission, mgxs.chi)
    if any(value is not None for value in fission_fields) and not all(
        value is not None for value in fission_fields
    ):
        raise ValueError("fissionable MGXS requires fission, nu_fission, and chi")
    fissionable = all(value is not None for value in fission_fields)

    output = None if output_directory is None else Path(output_directory)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    figures = {}
    stem = _filename_stem(mgxs.logical_domain)

    # All reaction-rate vectors below share macroscopic cross-section units and
    # can therefore be compared on one physical-energy axis.
    figure, axis = plt.subplots()
    axis.stairs(mgxs.total, mgxs.energy_bounds_ev, label="Total")
    axis.stairs(mgxs.absorption, mgxs.energy_bounds_ev, label="Absorption")
    if fissionable:
        axis.stairs(mgxs.fission, mgxs.energy_bounds_ev, label="Fission")
        axis.stairs(mgxs.nu_fission, mgxs.energy_bounds_ev, label="Nu-fission")
    axis.set_xscale("log")
    axis.set_xlabel("Energy [eV]")
    axis.set_ylabel("Cross section [cm^-1]")
    axis.set_title(f"{mgxs.logical_domain} MGXS cross sections")
    axis.legend()
    figure.tight_layout()
    figures["cross_sections"] = figure

    if fissionable:
        # Solver-ready chi is dimensionless, so it must not share the reaction
        # cross-section axis above.
        figure, axis = plt.subplots()
        axis.stairs(mgxs.chi, mgxs.energy_bounds_ev)
        axis.set_xscale("log")
        axis.set_xlabel("Energy [eV]")
        axis.set_ylabel("Chi")
        axis.set_title(f"{mgxs.logical_domain} fission spectrum")
        figure.tight_layout()
        figures["chi"] = figure

    for moment in moments:
        figures[f"scatter_p{moment}"] = _plot_group_matrix(
            mgxs.scatter[moment],
            mgxs.energy_bounds_ev,
            title=f"{mgxs.logical_domain} P{moment} scattering",
            colorbar_label="Scattering cross section [cm^-1]",
        )

    if fissionable:
        # This is a derived production operator, not a matrix stored directly
        # by OpenMC: rows are incident groups and columns are destination groups.
        fission_matrix = mgxs.nu_fission[:, None] * mgxs.chi[None, :]
        figures["fission_matrix"] = _plot_group_matrix(
            fission_matrix,
            mgxs.energy_bounds_ev,
            title=f"{mgxs.logical_domain} derived fission production",
            colorbar_label="Fission production [cm^-1]",
        )

    if output is not None:
        filenames = {
            "cross_sections": f"{stem}_cross_sections.png",
            "chi": f"{stem}_chi.png",
            "fission_matrix": f"{stem}_fission_matrix.png",
        }
        for moment in moments:
            filenames[f"scatter_p{moment}"] = f"{stem}_scatter_p{moment}.png"

        for name, figure in figures.items():
            figure.savefig(output / filenames[name])

    if show:
        plt.show()

    return figures
