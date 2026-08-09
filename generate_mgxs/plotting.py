"""Headless-friendly plots for spectra and multigroup cross sections."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .mgxs import MGXS


def plot_spectra(
    openmc,
    opensn,
    direct,
    *,
    output_directory=None,
    show=False,
):
    """Plot three complementary OpenMC/OpenSn/direct spectrum comparisons."""
    import matplotlib.pyplot as plt

    bounds = np.asarray(direct.energy_bounds_ev)
    for label, spectrum in (("OpenMC", openmc), ("OpenSn", opensn)):
        if not np.array_equal(spectrum.energy_bounds_ev, bounds):
            raise ValueError(f"{label} and Direct must have identical energy bounds")

    # Geometric group centers are appropriate marker locations on the common
    # logarithmic energy axis; stairs() still shows the group-integrated bins.
    centers = np.sqrt(bounds[:-1] * bounds[1:])
    normalized = {
        "OpenMC": openmc.normalized,
        "OpenSn": opensn.normalized,
        "Direct": direct.normalized,
    }
    styles = {
        "OpenMC": ("C0", "o"),
        "OpenSn": ("C1", "s"),
        "Direct": ("C2", "^"),
    }

    figures = {}

    # --- Normalized group-integrated spectrum ----------------------------
    figure, axis = plt.subplots()
    for label, values in normalized.items():
        color, marker = styles[label]
        axis.stairs(values, bounds, color=color, label=label)
        axis.plot(
            centers,
            values,
            color=color,
            marker=marker,
            linestyle="none",
            label="_nolegend_",
        )

    if openmc.std_dev is not None:
        # Normalize sigma by the same scalar flux sum as the mean. This ignores
        # covariance between groups, so the band is a diagnostic rather than a
        # propagated uncertainty on the normalized distribution.
        normalized_sigma = openmc.std_dev / np.sum(openmc.values)
        lower = normalized["OpenMC"] - normalized_sigma
        upper = normalized["OpenMC"] + normalized_sigma
        axis.stairs(
            upper,
            bounds,
            baseline=lower,
            fill=True,
            color=styles["OpenMC"][0],
            alpha=0.2,
            label="OpenMC ±1σ (covariance ignored)",
        )

    axis.set_xscale("log")
    axis.set_xlabel("Energy [eV]")
    axis.set_ylabel("Normalized group-integrated flux")
    axis.legend()
    figure.tight_layout()
    figures["group_spectrum"] = figure

    # --- Spectrum per unit lethargy --------------------------------------
    lethargy_width = np.log(bounds[1:] / bounds[:-1])
    figure, axis = plt.subplots()
    for label, values in normalized.items():
        per_lethargy = values / lethargy_width
        color, marker = styles[label]
        axis.stairs(per_lethargy, bounds, color=color, label=label)
        axis.plot(
            centers,
            per_lethargy,
            color=color,
            marker=marker,
            linestyle="none",
            label="_nolegend_",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Energy [eV]")
    axis.set_ylabel("Normalized flux per unit lethargy")
    axis.legend()
    figure.tight_layout()
    figures["flux_per_lethargy"] = figure

    # --- Relative differences from the direct balance solution -----------
    direct_values = normalized["Direct"]
    floor = max(1.0e-14, 1.0e-6 * float(np.max(np.abs(direct_values))))
    denominator = np.maximum(np.abs(direct_values), floor)

    figure, axis = plt.subplots()
    for label in ("OpenMC", "OpenSn"):
        difference = (normalized[label] - direct_values) / denominator
        color, marker = styles[label]
        axis.stairs(difference, bounds, color=color, label=f"{label} vs Direct")
        axis.plot(
            centers,
            difference,
            color=color,
            marker=marker,
            linestyle="none",
            label="_nolegend_",
        )

    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("Energy [eV]")
    axis.set_ylabel("Relative difference from Direct")
    axis.legend()
    figure.tight_layout()
    figures["relative_differences"] = figure

    if output_directory is not None:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(output / f"{name}.png")

    if show:
        plt.show()

    return figures


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

    # Matrix displays follow the transport-review convention: incoming energy
    # decreases from left to right, while outgoing energy increases upward.
    # Invert only the x axis; the canonical [g_in, g_out] data are untouched.
    axis.invert_xaxis()
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
