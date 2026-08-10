"""Headless-friendly plots for spectra and multigroup cross sections."""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np

from .mgxs import MGXS


_LOG_PLOT_MINIMUM_EV = 1.0e-5


def _plot_energy_bounds(energy_bounds_ev):
    """Return physical bounds suitable for an ordinary logarithmic axis."""
    bounds = np.asarray(energy_bounds_ev)
    if bounds[0] != 0.0:
        return bounds

    # Zero is a valid lower edge for the first physical group but cannot be
    # represented on a logarithmic axis. Move only its displayed edge, leaving
    # the scientific group definition and all group values untouched.
    plotting_bounds = bounds.copy()
    plotting_bounds[0] = _LOG_PLOT_MINIMUM_EV
    warnings.warn(
        "Lowest energy boundary is 0 eV; using 1e-5 eV for logarithmic "
        "plotting only.",
        UserWarning,
        stacklevel=2,
    )

    return plotting_bounds


def plot_spectra(
    openmc,
    opensn,
    direct,
    *,
    include=("openmc", "opensn", "direct"),
    output_directory=None,
    show=False,
):
    """Plot selected OpenMC/OpenSn/direct spectrum comparisons."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    solver_results = {
        "openmc": ("OpenMC", openmc),
        "opensn": ("OpenSn", opensn),
        "direct": ("Direct", direct),
    }
    if isinstance(include, (str, bytes)):
        raise ValueError("include must be an iterable of solver names")
    try:
        requested = tuple(include)
    except TypeError as error:
        raise ValueError("include must be an iterable of solver names") from error

    if not requested:
        raise ValueError("include must select at least one solver")
    if any(not isinstance(name, str) for name in requested):
        raise ValueError("included solver names must be strings")
    if len(set(requested)) != len(requested):
        raise ValueError("include must not contain duplicate solver names")
    unknown = set(requested) - set(solver_results)
    if unknown:
        raise ValueError(f"unknown included solver {sorted(unknown)[0]!r}")

    # Canonical ordering keeps colors and legends stable even when include is
    # supplied as an unordered collection.
    included = {}
    for name, (label, spectrum) in solver_results.items():
        if name not in requested:
            continue
        if spectrum is None:
            raise ValueError(f"{label} result was requested but is None")
        included[label] = spectrum

    reference_label, reference = next(iter(included.items()))
    physical_bounds = np.asarray(reference.energy_bounds_ev)
    for label, spectrum in tuple(included.items())[1:]:
        if not np.array_equal(spectrum.energy_bounds_ev, physical_bounds):
            raise ValueError(
                f"{label} and {reference_label} must have identical energy bounds"
            )

    plotting_bounds = _plot_energy_bounds(physical_bounds)

    # Widths and midpoint energies are physical quantities. They must use the
    # true boundaries even when a zero edge is shifted for log-axis display.
    energy_widths = np.diff(physical_bounds)
    energy_midpoints = physical_bounds[:-1] + 0.5 * energy_widths
    normalized = {
        label: spectrum.normalized
        for label, spectrum in included.items()
    }
    energy_spectra = {
        label: values / energy_widths
        for label, values in normalized.items()
    }
    lethargy_spectra = {
        label: energy_midpoints * values
        for label, values in energy_spectra.items()
    }
    styles = {
        "OpenMC": ("C0", "o"),
        "OpenSn": ("C1", "s"),
        "Direct": ("C2", "^"),
    }
    marker_stride = max(1, energy_widths.size // 20)

    energy_sigma = lethargy_sigma = None
    if "OpenMC" in included and included["OpenMC"].std_dev is not None:
        # Normalize sigma by the same scalar flux sum as the mean. This ignores
        # covariance between groups, so both bands are diagnostics rather than
        # propagated uncertainty on the normalized distributions.
        openmc_result = included["OpenMC"]
        normalized_sigma = openmc_result.std_dev / np.sum(openmc_result.values)
        energy_sigma = normalized_sigma / energy_widths
        lethargy_sigma = energy_midpoints * energy_sigma

    figures = {}

    # --- Normalized flux spectrum per unit energy ------------------------
    figure, axis = plt.subplots()
    legend_handles = []
    for label, values in energy_spectra.items():
        color, marker = styles[label]
        values_plot = np.insert(values, 0, values[0])
        axis.loglog(
            plotting_bounds,
            values_plot,
            color=color,
            drawstyle="steps",
            label=label,
        )
        axis.plot(
            energy_midpoints,
            values,
            color=color,
            marker=marker,
            markevery=marker_stride,
            markersize=3.25,
            linestyle="none",
            label="_nolegend_",
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markersize=3.25,
                label=label,
            )
        )

    if energy_sigma is not None:
        lower = energy_spectra["OpenMC"] - energy_sigma
        upper = energy_spectra["OpenMC"] + energy_sigma
        lower_plot = np.insert(lower, 0, lower[0])
        upper_plot = np.insert(upper, 0, upper[0])
        axis.fill_between(
            plotting_bounds,
            lower_plot,
            upper_plot,
            step="pre",
            color=styles["OpenMC"][0],
            alpha=0.2,
            label="OpenMC ±1σ (covariance ignored)",
        )
        legend_handles.append(
            Patch(
                facecolor=styles["OpenMC"][0],
                alpha=0.2,
                label="OpenMC ±1σ (covariance ignored)",
            )
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Energy [eV]")
    axis.set_ylabel("Normalized flux spectrum [1/eV]")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(handles=legend_handles)
    figure.tight_layout()
    figures["flux_spectrum"] = figure

    # --- Midpoint representation of normalized E * phi(E) ---------------
    figure, axis = plt.subplots()
    legend_handles = []
    for label, values in lethargy_spectra.items():
        color, marker = styles[label]
        values_plot = np.insert(values, 0, values[0])
        axis.semilogx(
            plotting_bounds,
            values_plot,
            color=color,
            drawstyle="steps",
            label=label,
        )
        axis.plot(
            energy_midpoints,
            values,
            color=color,
            marker=marker,
            markevery=marker_stride,
            markersize=3.25,
            linestyle="none",
            label="_nolegend_",
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markersize=3.25,
                label=label,
            )
        )

    if lethargy_sigma is not None:
        lower = lethargy_spectra["OpenMC"] - lethargy_sigma
        upper = lethargy_spectra["OpenMC"] + lethargy_sigma
        lower_plot = np.insert(lower, 0, lower[0])
        upper_plot = np.insert(upper, 0, upper[0])
        axis.fill_between(
            plotting_bounds,
            lower_plot,
            upper_plot,
            step="pre",
            color=styles["OpenMC"][0],
            alpha=0.2,
            label="OpenMC ±1σ (covariance ignored)",
        )
        legend_handles.append(
            Patch(
                facecolor=styles["OpenMC"][0],
                alpha=0.2,
                label="OpenMC ±1σ (covariance ignored)",
            )
        )

    axis.set_xscale("log")
    axis.set_yscale("linear")
    axis.set_xlabel("Energy [eV]")
    axis.set_ylabel("Normalized flux per unit lethargy")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(handles=legend_handles)
    figure.tight_layout()
    figures["lethargy_spectrum"] = figure

    # --- Relative differences from the direct balance solution -----------
    # The common group width cancels when two per-energy spectra are compared
    # group by group, so normalized integrated values give the same ratio.
    comparison_labels = [label for label in normalized if label != "Direct"]
    if "Direct" in normalized and comparison_labels:
        direct_values = normalized["Direct"]
        floor = max(1.0e-14, 1.0e-6 * float(np.max(np.abs(direct_values))))
        denominator = np.maximum(np.abs(direct_values), floor)

        figure, axis = plt.subplots()
        for label in comparison_labels:
            difference = (normalized[label] - direct_values) / denominator
            color, marker = styles[label]
            axis.stairs(
                difference,
                plotting_bounds,
                color=color,
                label=f"{label} vs Direct",
            )
            axis.plot(
                energy_midpoints,
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


def _positive_magnitudes(values):
    """Replace nonpositive plotting values with NaN in an independent copy."""
    plotted = np.array(values, dtype=float, copy=True)
    plotted[plotted <= 0.0] = np.nan
    return plotted


def _plot_group_matrix(
    values_gin_gout,
    *,
    title: str,
    colorbar_label: str,
    logarithmic_color=False,
):
    """Plot a canonical ``[g_in, g_out]`` matrix in high-to-low group order."""
    import matplotlib.pyplot as plt

    scientific_values = np.asarray(values_gin_gout)
    if (
        scientific_values.ndim != 2
        or scientific_values.shape[0] != scientific_values.shape[1]
    ):
        raise ValueError("group matrix must be square")

    groups = scientific_values.shape[0]
    group_edges = np.arange(0.5, groups + 1.5)

    # Canonical arrays are ascending in physical energy, whereas conventional
    # multigroup index 1 denotes the highest-energy group. Reverse both energy
    # axes in this plotting view only. The transpose maps incoming groups to
    # Matplotlib's horizontal dimension and outgoing groups to its vertical one.
    displayed_values = scientific_values[::-1, ::-1].T

    norm = None
    if logarithmic_color:
        from matplotlib.colors import LogNorm

        if np.any(scientific_values < 0.0):
            raise ValueError("logarithmic matrix color requires nonnegative values")

        # Mask nonpositive cells instead of taking log(data). The colorbar thus
        # retains physical cross-section units and zero cells render blank.
        displayed_values = np.ma.masked_less_equal(displayed_values, 0.0)
        positive = scientific_values[scientific_values > 0.0]
        if positive.size:
            norm = LogNorm(
                vmin=float(np.min(positive)),
                vmax=float(np.max(positive)),
            )
            colorbar_label = f"{colorbar_label} (log scale)"

    figure, axis = plt.subplots()
    mesh = axis.pcolormesh(
        group_edges,
        group_edges,
        displayed_values,
        shading="flat",
        norm=norm,
    )

    # Group number increases left-to-right for incoming groups and
    # top-to-bottom for outgoing groups, placing high/high at top-left and
    # low/low at bottom-right.
    axis.invert_yaxis()
    axis.set_xlabel("Incoming group")
    axis.set_ylabel("Outgoing group")
    axis.set_title(title)

    if groups <= 12:
        ticks = np.arange(1, groups + 1)
    else:
        ticks = np.unique(np.rint(np.linspace(1, groups, 9)).astype(int))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)

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
    plotting_bounds = _plot_energy_bounds(mgxs.energy_bounds_ev)

    # All reaction-rate vectors below share macroscopic cross-section units and
    # can therefore be compared on one physical-energy axis.
    figure, axis = plt.subplots()
    vectors = [
        ("Total", mgxs.total),
        ("Absorption", mgxs.absorption),
    ]
    if fissionable:
        vectors.extend(
            (
                ("Fission", mgxs.fission),
                ("Nu-fission", mgxs.nu_fission),
            )
        )
    for label, values in vectors:
        values_plot = _positive_magnitudes(
            np.insert(values, 0, values[0])
        )
        axis.semilogx(
            plotting_bounds,
            values_plot,
            drawstyle="steps",
            linewidth=1,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
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
        chi_plot = np.append(mgxs.chi, mgxs.chi[-1])
        axis.plot(
            plotting_bounds,
            chi_plot,
            drawstyle="steps-post",
            linewidth=1,
        )
        axis.set_xscale("log")
        axis.set_xlabel("Energy [eV]")
        axis.set_ylabel("Chi")
        axis.set_title(f"{mgxs.logical_domain} fission spectrum")
        figure.tight_layout()
        figures["chi"] = figure

    for moment in moments:
        scatter = mgxs.scatter[moment]
        figures[f"scatter_p{moment}"] = _plot_group_matrix(
            scatter,
            title=f"{mgxs.logical_domain} P{moment} scattering",
            colorbar_label="Scattering cross section [cm^-1]",
            logarithmic_color=(moment == 0 and np.all(scatter >= 0.0)),
        )

    if fissionable:
        # This is a derived production operator, not a matrix stored directly
        # by OpenMC: rows are incident groups and columns are destination groups.
        fission_matrix = mgxs.nu_fission[:, None] * mgxs.chi[None, :]
        figures["fission_matrix"] = _plot_group_matrix(
            fission_matrix,
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
